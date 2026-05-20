//! Query context.

use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fmt::{self, Debug, Display, Formatter},
    ops::Deref,
    sync::{Arc, Mutex},
};

use async_graphql_parser::types::ConstDirective;
use async_graphql_value::{Value as InputValue, Variables};
use fnv::FnvHashMap;
use serde::{
    Serialize,
    ser::{SerializeSeq, Serializer},
};

use crate::{
    Error, InputType, Lookahead, Name, OneofObjectType, PathSegment, Pos, Positioned, Result,
    ServerError, ServerResult, UploadValue, Value,
    extensions::Extensions,
    parser::types::{
        Directive, Field, FragmentDefinition, OperationDefinition, Selection, SelectionSet,
    },
    schema::{IntrospectionMode, SchemaEnv},
};

/// Data related functions of the context.
pub trait DataContext<'a> {
    /// Gets the global data defined in the `Context` or `Schema`.
    ///
    /// If both `Schema` and `Query` have the same data type, the data in the
    /// `Query` is obtained.
    ///
    /// # Errors
    ///
    /// Returns a `Error` if the specified type data does not exist.
    fn data<D: Any + Send + Sync>(&self) -> Result<&'a D>;

    /// Gets the global data defined in the `Context` or `Schema`.
    ///
    /// # Panics
    ///
    /// It will panic if the specified data type does not exist.
    fn data_unchecked<D: Any + Send + Sync>(&self) -> &'a D;

    /// Gets the global data defined in the `Context` or `Schema` or `None` if
    /// the specified type data does not exist.
    fn data_opt<D: Any + Send + Sync>(&self) -> Option<&'a D>;
}

/// Schema/Context data.
///
/// This is a type map, allowing you to store anything inside it.
#[derive(Default)]
pub struct Data(FnvHashMap<TypeId, Box<dyn Any + Sync + Send>>);

impl Deref for Data {
    type Target = FnvHashMap<TypeId, Box<dyn Any + Sync + Send>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Data {
    /// Insert data.
    pub fn insert<D: Any + Send + Sync>(&mut self, data: D) {
        self.0.insert(TypeId::of::<D>(), Box::new(data));
    }

    pub(crate) fn merge(&mut self, other: Data) {
        self.0.extend(other.0);
    }
}

impl Debug for Data {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.debug_tuple("Data").finish()
    }
}

/// Context for `SelectionSet`
pub type ContextSelectionSet<'a> = ContextBase<'a, &'a Positioned<SelectionSet>>;

/// Context object for resolve field
pub type Context<'a> = ContextBase<'a, &'a Positioned<Field>>;

/// Context object for execute directive.
pub type ContextDirective<'a> = ContextBase<'a, &'a Positioned<Directive>>;

/// A segment in the path to the current query.
///
/// This is a borrowed form of [`PathSegment`](enum.PathSegment.html) used
/// during execution instead of passed back when errors occur.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(untagged)]
pub enum QueryPathSegment<'a> {
    /// We are currently resolving an element in a list.
    Index(usize),
    /// We are currently resolving a field in an object.
    Name(&'a str),
}

/// A path to the current query.
///
/// The path is stored as a kind of reverse linked list.
#[derive(Debug, Clone, Copy)]
pub struct QueryPathNode<'a> {
    /// The parent node to this, if there is one.
    pub parent: Option<&'a QueryPathNode<'a>>,

    /// The current path segment being resolved.
    pub segment: QueryPathSegment<'a>,
}

impl serde::Serialize for QueryPathNode<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut seq = serializer.serialize_seq(None)?;
        self.try_for_each(|segment| seq.serialize_element(segment))?;
        seq.end()
    }
}

impl Display for QueryPathNode<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut first = true;
        self.try_for_each(|segment| {
            if !first {
                write!(f, ".")?;
            }
            first = false;

            match segment {
                QueryPathSegment::Index(idx) => write!(f, "{}", *idx),
                QueryPathSegment::Name(name) => write!(f, "{}", name),
            }
        })
    }
}

impl<'a> QueryPathNode<'a> {
    /// Get the current field name.
    ///
    /// This traverses all the parents of the node until it finds one that is a
    /// field name.
    pub fn field_name(&self) -> &str {
        std::iter::once(self)
            .chain(self.parents())
            .find_map(|node| match node.segment {
                QueryPathSegment::Name(name) => Some(name),
                QueryPathSegment::Index(_) => None,
            })
            .unwrap()
    }

    /// Get the path represented by `Vec<String>`; numbers will be stringified.
    #[must_use]
    pub fn to_string_vec(self) -> Vec<String> {
        let mut res = Vec::new();
        self.for_each(|s| {
            res.push(match s {
                QueryPathSegment::Name(name) => (*name).to_string(),
                QueryPathSegment::Index(idx) => idx.to_string(),
            });
        });
        res
    }

    /// Iterate over the parents of the node.
    pub fn parents(&self) -> Parents<'_> {
        Parents(self)
    }

    pub(crate) fn for_each<F: FnMut(&QueryPathSegment<'a>)>(&self, mut f: F) {
        let _ = self.try_for_each::<std::convert::Infallible, _>(|segment| {
            f(segment);
            Ok(())
        });
    }

    pub(crate) fn try_for_each<E, F: FnMut(&QueryPathSegment<'a>) -> Result<(), E>>(
        &self,
        mut f: F,
    ) -> Result<(), E> {
        self.try_for_each_ref(&mut f)
    }

    fn try_for_each_ref<E, F: FnMut(&QueryPathSegment<'a>) -> Result<(), E>>(
        &self,
        f: &mut F,
    ) -> Result<(), E> {
        if let Some(parent) = &self.parent {
            parent.try_for_each_ref(f)?;
        }
        f(&self.segment)
    }
}

/// An iterator over the parents of a
/// [`QueryPathNode`](struct.QueryPathNode.html).
#[derive(Debug, Clone)]
pub struct Parents<'a>(&'a QueryPathNode<'a>);

impl<'a> Parents<'a> {
    /// Get the current query path node, which the next call to `next` will get
    /// the parents of.
    #[must_use]
    pub fn current(&self) -> &'a QueryPathNode<'a> {
        self.0
    }
}

impl<'a> Iterator for Parents<'a> {
    type Item = &'a QueryPathNode<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let parent = self.0.parent;
        if let Some(parent) = parent {
            self.0 = parent;
        }
        parent
    }
}

impl std::iter::FusedIterator for Parents<'_> {}

/// Query context.
///
/// **This type is not stable and should not be used directly.**
#[derive(Clone)]
pub struct ContextBase<'a, T> {
    /// The current path node being resolved.
    pub path_node: Option<QueryPathNode<'a>>,
    /// If `true` means the current field is for introspection.
    pub(crate) is_for_introspection: bool,
    #[doc(hidden)]
    pub item: T,
    #[doc(hidden)]
    pub schema_env: &'a SchemaEnv,
    #[doc(hidden)]
    pub query_env: &'a QueryEnv,
    #[doc(hidden)]
    pub execute_data: Option<&'a Data>,
}

#[doc(hidden)]
pub struct QueryEnvInner {
    pub extensions: Extensions,
    pub variables: Variables,
    pub operation_name: Option<String>,
    pub operation: Positioned<OperationDefinition>,
    pub fragments: HashMap<Name, Positioned<FragmentDefinition>>,
    pub uploads: Vec<UploadValue>,
    pub session_data: Arc<Data>,
    pub query_data: Arc<Data>,
    pub http_headers: Mutex<http::HeaderMap>,
    pub introspection_mode: IntrospectionMode,
    pub errors: Mutex<Vec<ServerError>>,
}

#[doc(hidden)]
#[derive(Clone)]
pub struct QueryEnv(Arc<QueryEnvInner>);

impl Deref for QueryEnv {
    type Target = QueryEnvInner;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl QueryEnv {
    #[doc(hidden)]
    pub fn new(inner: QueryEnvInner) -> QueryEnv {
        QueryEnv(Arc::new(inner))
    }

    #[doc(hidden)]
    pub fn create_context<'a, T>(
        &'a self,
        schema_env: &'a SchemaEnv,
        path_node: Option<QueryPathNode<'a>>,
        item: T,
        execute_data: Option<&'a Data>,
    ) -> ContextBase<'a, T> {
        ContextBase {
            path_node,
            is_for_introspection: false,
            item,
            schema_env,
            query_env: self,
            execute_data,
        }
    }
}

impl<'a, T> DataContext<'a> for ContextBase<'a, T> {
    fn data<D: Any + Send + Sync>(&self) -> Result<&'a D> {
        ContextBase::data::<D>(self)
    }

    fn data_unchecked<D: Any + Send + Sync>(&self) -> &'a D {
        ContextBase::data_unchecked::<D>(self)
    }

    fn data_opt<D: Any + Send + Sync>(&self) -> Option<&'a D> {
        ContextBase::data_opt::<D>(self)
    }
}

impl<'a, T> ContextBase<'a, T> {
    #[doc(hidden)]
    pub fn with_field(
        &'a self,
        field: &'a Positioned<Field>,
    ) -> ContextBase<'a, &'a Positioned<Field>> {
        ContextBase {
            path_node: Some(QueryPathNode {
                parent: self.path_node.as_ref(),
                segment: QueryPathSegment::Name(&field.node.response_key().node),
            }),
            is_for_introspection: self.is_for_introspection,
            item: field,
            schema_env: self.schema_env,
            query_env: self.query_env,
            execute_data: self.execute_data,
        }
    }

    #[doc(hidden)]
    pub fn with_selection_set(
        &self,
        selection_set: &'a Positioned<SelectionSet>,
    ) -> ContextBase<'a, &'a Positioned<SelectionSet>> {
        ContextBase {
            path_node: self.path_node,
            is_for_introspection: self.is_for_introspection,
            item: selection_set,
            schema_env: self.schema_env,
            query_env: self.query_env,
            execute_data: self.execute_data,
        }
    }

    #[doc(hidden)]
    pub fn set_error_path(&self, error: ServerError) -> ServerError {
        if let Some(node) = self.path_node {
            let mut path = Vec::new();
            node.for_each(|current_node| {
                path.push(match current_node {
                    QueryPathSegment::Name(name) => PathSegment::Field((*name).to_string()),
                    QueryPathSegment::Index(idx) => PathSegment::Index(*idx),
                })
            });
            ServerError { path, ..error }
        } else {
            error
        }
    }

    /// Report a resolver error.
    ///
    /// When implementing `OutputType`, if an error occurs, call this function
    /// to report this error and return `Value::Null`.
    pub fn add_error(&self, error: ServerError) {
        self.query_env.errors.lock().unwrap().push(error);
    }

    /// Gets the global data defined in the `Context` or `Schema`.
    ///
    /// If both `Schema` and `Query` have the same data type, the data in the
    /// `Query` is obtained.
    ///
    /// # Errors
    ///
    /// Returns a `Error` if the specified type data does not exist.
    pub fn data<D: Any + Send + Sync>(&self) -> Result<&'a D> {
        self.data_opt::<D>().ok_or_else(|| {
            Error::new(format!(
                "Data `{}` does not exist.",
                std::any::type_name::<D>()
            ))
        })
    }

    /// Gets the global data defined in the `Context` or `Schema`.
    ///
    /// # Panics
    ///
    /// It will panic if the specified data type does not exist.
    pub fn data_unchecked<D: Any + Send + Sync>(&self) -> &'a D {
        self.data_opt::<D>()
            .unwrap_or_else(|| panic!("Data `{}` does not exist.", std::any::type_name::<D>()))
    }

    /// Gets the global data defined in the `Context` or `Schema` or `None` if
    /// the specified type data does not exist.
    pub fn data_opt<D: Any + Send + Sync>(&self) -> Option<&'a D> {
        self.execute_data
            .as_ref()
            .and_then(|execute_data| execute_data.get(&TypeId::of::<D>()))
            .or_else(|| self.query_env.query_data.0.get(&TypeId::of::<D>()))
            .or_else(|| self.query_env.session_data.0.get(&TypeId::of::<D>()))
            .or_else(|| self.schema_env.data.0.get(&TypeId::of::<D>()))
            .and_then(|d| d.downcast_ref::<D>())
    }

    /// Returns whether the HTTP header `key` is currently set on the response
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ::http::header::ACCESS_CONTROL_ALLOW_ORIGIN;
    /// use async_graphql::*;
    ///
    /// struct Query;
    ///
    /// #[Object]
    /// impl Query {
    ///     async fn greet(&self, ctx: &Context<'_>) -> String {
    ///         let header_exists = ctx.http_header_contains("Access-Control-Allow-Origin");
    ///         assert!(!header_exists);
    ///
    ///         ctx.insert_http_header(ACCESS_CONTROL_ALLOW_ORIGIN, "*");
    ///
    ///         let header_exists = ctx.http_header_contains("Access-Control-Allow-Origin");
    ///         assert!(header_exists);
    ///
    ///         String::from("Hello world")
    ///     }
    /// }
    /// ```
    pub fn http_header_contains(&self, key: impl http::header::AsHeaderName) -> bool {
        self.query_env
            .http_headers
            .lock()
            .unwrap()
            .contains_key(key)
    }

    /// Sets a HTTP header to response.
    ///
    /// If the header was not currently set on the response, then `None` is
    /// returned.
    ///
    /// If the response already contained this header then the new value is
    /// associated with this key and __all the previous values are
    /// removed__, however only a the first previous value is returned.
    ///
    /// See [`http::HeaderMap`] for more details on the underlying
    /// implementation
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ::http::{HeaderValue, header::ACCESS_CONTROL_ALLOW_ORIGIN};
    /// use async_graphql::*;
    ///
    /// struct Query;
    ///
    /// #[Object]
    /// impl Query {
    ///     async fn greet(&self, ctx: &Context<'_>) -> String {
    ///         // Headers can be inserted using the `http` constants
    ///         let was_in_headers = ctx.insert_http_header(ACCESS_CONTROL_ALLOW_ORIGIN, "*");
    ///         assert_eq!(was_in_headers, None);
    ///
    ///         // They can also be inserted using &str
    ///         let was_in_headers = ctx.insert_http_header("Custom-Header", "1234");
    ///         assert_eq!(was_in_headers, None);
    ///
    ///         // If multiple headers with the same key are `inserted` then the most recent
    ///         // one overwrites the previous. If you want multiple headers for the same key, use
    ///         // `append_http_header` for subsequent headers
    ///         let was_in_headers = ctx.insert_http_header("Custom-Header", "Hello World");
    ///         assert_eq!(was_in_headers, Some(HeaderValue::from_static("1234")));
    ///
    ///         String::from("Hello world")
    ///     }
    /// }
    /// ```
    pub fn insert_http_header(
        &self,
        name: impl http::header::IntoHeaderName,
        value: impl TryInto<http::HeaderValue>,
    ) -> Option<http::HeaderValue> {
        if let Ok(value) = value.try_into() {
            self.query_env
                .http_headers
                .lock()
                .unwrap()
                .insert(name, value)
        } else {
            None
        }
    }

    /// Sets a HTTP header to response.
    ///
    /// If the header was not currently set on the response, then `false` is
    /// returned.
    ///
    /// If the response did have this header then the new value is appended to
    /// the end of the list of values currently associated with the key,
    /// however the key is not updated _(which is important for types that
    /// can be `==` without being identical)_.
    ///
    /// See [`http::HeaderMap`] for more details on the underlying
    /// implementation
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use ::http::header::SET_COOKIE;
    /// use async_graphql::*;
    ///
    /// struct Query;
    ///
    /// #[Object]
    /// impl Query {
    ///     async fn greet(&self, ctx: &Context<'_>) -> String {
    ///         // Insert the first instance of the header
    ///         ctx.insert_http_header(SET_COOKIE, "Chocolate Chip");
    ///
    ///         // Subsequent values should be appended
    ///         let header_already_exists = ctx.append_http_header("Set-Cookie", "Macadamia");
    ///         assert!(header_already_exists);
    ///
    ///         String::from("Hello world")
    ///     }
    /// }
    /// ```
    pub fn append_http_header(
        &self,
        name: impl http::header::IntoHeaderName,
        value: impl TryInto<http::HeaderValue>,
    ) -> bool {
        if let Ok(value) = value.try_into() {
            self.query_env
                .http_headers
                .lock()
                .unwrap()
                .append(name, value)
        } else {
            false
        }
    }

    fn var_value(&self, name: &str, pos: Pos) -> ServerResult<Value> {
        self.query_env
            .operation
            .node
            .variable_definitions
            .iter()
            .find(|def| def.node.name.node == name)
            .and_then(|def| {
                self.query_env
                    .variables
                    .get(&def.node.name.node)
                    .or_else(|| def.node.default_value())
            })
            .cloned()
            .ok_or_else(|| {
                ServerError::new(format!("Variable {} is not defined.", name), Some(pos))
            })
    }

    pub(crate) fn resolve_input_value(&self, value: Positioned<InputValue>) -> ServerResult<Value> {
        let pos = value.pos;
        value
            .node
            .into_const_with(|name| self.var_value(&name, pos))
    }

    #[doc(hidden)]
    fn get_param_value<Q: InputType>(
        &self,
        arguments: &[(Positioned<Name>, Positioned<InputValue>)],
        name: &str,
        default: Option<fn() -> Q>,
    ) -> ServerResult<(Pos, Q)> {
        let value = arguments
            .iter()
            .find(|(n, _)| n.node.as_str() == name)
            .map(|(_, value)| value)
            .cloned();
        if value.is_none()
            && let Some(default) = default
        {
            return Ok((Pos::default(), default()));
        }
        let (pos, value) = match value {
            Some(value) => (value.pos, Some(self.resolve_input_value(value)?)),
            None => (Pos::default(), None),
        };
        InputType::parse(value)
            .map(|value| (pos, value))
            .map_err(|e| e.into_server_error(pos))
    }

    #[doc(hidden)]
    #[must_use]
    pub fn with_index(&'a self, idx: usize) -> ContextBase<'a, T>
    where
        T: Copy,
    {
        ContextBase {
            path_node: Some(QueryPathNode {
                parent: self.path_node.as_ref(),
                segment: QueryPathSegment::Index(idx),
            }),
            is_for_introspection: self.is_for_introspection,
            item: self.item,
            schema_env: self.schema_env,
            query_env: self.query_env,
            execute_data: self.execute_data,
        }
    }
}

impl<'a> ContextBase<'a, &'a Positioned<Field>> {
    #[doc(hidden)]
    pub fn param_value<T: InputType>(
        &self,
        name: &str,
        default: Option<fn() -> T>,
    ) -> ServerResult<(Pos, T)> {
        self.get_param_value(&self.item.node.arguments, name, default)
    }

    #[doc(hidden)]
    pub fn oneof_param_value<T: OneofObjectType>(&self) -> ServerResult<(Pos, T)> {
        use indexmap::IndexMap;

        let mut map = IndexMap::new();

        for (name, value) in &self.item.node.arguments {
            let value = self.resolve_input_value(value.clone())?;
            map.insert(name.node.clone(), value);
        }

        InputType::parse(Some(Value::Object(map)))
            .map(|value| (self.item.pos, value))
            .map_err(|e| e.into_server_error(self.item.pos))
    }

    /// Creates a uniform interface to inspect the forthcoming selections.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use async_graphql::*;
    ///
    /// #[derive(SimpleObject)]
    /// struct Detail {
    ///     c: i32,
    ///     d: i32,
    /// }
    ///
    /// #[derive(SimpleObject)]
    /// struct MyObj {
    ///     a: i32,
    ///     b: i32,
    ///     detail: Detail,
    /// }
    ///
    /// struct Query;
    ///
    /// #[Object]
    /// impl Query {
    ///     async fn obj(&self, ctx: &Context<'_>) -> MyObj {
    ///         if ctx.look_ahead().field("a").exists() {
    ///             // This is a query like `obj { a }`
    ///         } else if ctx.look_ahead().field("detail").field("c").exists() {
    ///             // This is a query like `obj { detail { c } }`
    ///         } else {
    ///             // This query doesn't have `a`
    ///         }
    ///         unimplemented!()
    ///     }
    /// }
    /// ```
    pub fn look_ahead(&self) -> Lookahead<'_> {
        Lookahead::new(&self.query_env.fragments, &self.item.node, self)
    }

    /// Get the current field.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use async_graphql::*;
    ///
    /// #[derive(SimpleObject)]
    /// struct MyObj {
    ///     a: i32,
    ///     b: i32,
    ///     c: i32,
    /// }
    ///
    /// pub struct Query;
    ///
    /// #[Object]
    /// impl Query {
    ///     async fn obj(&self, ctx: &Context<'_>) -> MyObj {
    ///         let fields = ctx
    ///             .field()
    ///             .selection_set()
    ///             .map(|field| field.name())
    ///             .collect::<Vec<_>>();
    ///         assert_eq!(fields, vec!["a", "b", "c"]);
    ///         MyObj { a: 1, b: 2, c: 3 }
    ///     }
    /// }
    ///
    /// # tokio::runtime::Runtime::new().unwrap().block_on(async move {
    /// let schema = Schema::new(Query, EmptyMutation, EmptySubscription);
    /// assert!(schema.execute("{ obj { a b c }}").await.is_ok());
    /// assert!(schema.execute("{ obj { a ... { b c } }}").await.is_ok());
    /// assert!(
    ///     schema
    ///         .execute("{ obj { a ... BC }} fragment BC on MyObj { b c }")
    ///         .await
    ///         .is_ok()
    /// );
    /// # });
    /// ```
    pub fn field(&self) -> SelectionField<'_> {
        SelectionField {
            fragments: &self.query_env.fragments,
            field: &self.item.node,
            context: self,
            type_condition: None,
        }
    }
}

impl<'a> ContextBase<'a, &'a Positioned<Directive>> {
    #[doc(hidden)]
    pub fn param_value<T: InputType>(
        &self,
        name: &str,
        default: Option<fn() -> T>,
    ) -> ServerResult<(Pos, T)> {
        self.get_param_value(&self.item.node.arguments, name, default)
    }
}

/// Selection field.
#[derive(Clone, Copy)]
pub struct SelectionField<'a> {
    pub(crate) fragments: &'a HashMap<Name, Positioned<FragmentDefinition>>,
    pub(crate) field: &'a Field,
    pub(crate) context: &'a Context<'a>,
    pub(crate) type_condition: Option<&'a str>,
}

impl<'a> SelectionField<'a> {
    /// Get the name of this field.
    #[inline]
    pub fn name(&self) -> &'a str {
        self.field.name.node.as_str()
    }

    /// Get the alias of this field.
    #[inline]
    pub fn alias(&self) -> Option<&'a str> {
        self.field.alias.as_ref().map(|alias| alias.node.as_str())
    }

    /// Get the directives of this field.
    pub fn directives(&self) -> ServerResult<Vec<ConstDirective>> {
        let mut directives = Vec::with_capacity(self.field.directives.len());

        for directive in &self.field.directives {
            let directive = &directive.node;

            let mut arguments = Vec::with_capacity(directive.arguments.len());
            for (name, value) in &directive.arguments {
                let pos = name.pos;
                arguments.push((
                    name.clone(),
                    value.position_node(
                        value
                            .node
                            .clone()
                            .into_const_with(|name| self.context.var_value(&name, pos))?,
                    ),
                ));
            }

            directives.push(ConstDirective {
                name: directive.name.clone(),
                arguments,
            });
        }

        Ok(directives)
    }

    /// Get the arguments of this field.
    pub fn arguments(&self) -> ServerResult<Vec<(Name, Value)>> {
        let mut arguments = Vec::with_capacity(self.field.arguments.len());
        for (name, value) in &self.field.arguments {
            let pos = name.pos;
            arguments.push((
                name.node.clone(),
                value
                    .clone()
                    .node
                    .into_const_with(|name| self.context.var_value(&name, pos))?,
            ));
        }
        Ok(arguments)
    }

    /// Get all subfields of the current selection set.
    pub fn selection_set(&self) -> impl Iterator<Item = SelectionField<'a>> {
        SelectionFieldsIter {
            fragments: self.fragments,
            iter: vec![(self.field.selection_set.node.items.iter(), None)],
            context: self.context,
        }
    }

    /// The fragment type condition under which this field was requested, if any.
    #[inline]
    pub fn type_condition(&self) -> Option<&'a str> {
        self.type_condition
    }
}

impl Debug for SelectionField<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        struct DebugSelectionSet<'a>(Vec<SelectionField<'a>>);

        impl Debug for DebugSelectionSet<'_> {
            fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
                f.debug_list().entries(&self.0).finish()
            }
        }

        f.debug_struct(self.name())
            .field("name", &self.name())
            .field(
                "selection_set",
                &DebugSelectionSet(self.selection_set().collect()),
            )
            .finish()
    }
}

type SelectionFrame<'a> = (
    std::slice::Iter<'a, Positioned<Selection>>,
    Option<&'a str>,
);

struct SelectionFieldsIter<'a> {
    fragments: &'a HashMap<Name, Positioned<FragmentDefinition>>,
    iter: Vec<SelectionFrame<'a>>,
    context: &'a Context<'a>,
}

impl<'a> Iterator for SelectionFieldsIter<'a> {
    type Item = SelectionField<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let (it, active_type_condition) = self.iter.last_mut()?;
            let active_type_condition = *active_type_condition;
            let item = it.next();

            match item {
                Some(selection) => match &selection.node {
                    Selection::Field(field) => {
                        return Some(SelectionField {
                            fragments: self.fragments,
                            field: &field.node,
                            context: self.context,
                            type_condition: active_type_condition,
                        });
                    }
                    Selection::FragmentSpread(fragment_spread) => {
                        if let Some(fragment) =
                            self.fragments.get(&fragment_spread.node.fragment_name.node)
                        {
                            self.iter.push((
                                fragment.node.selection_set.node.items.iter(),
                                Some(fragment.node.type_condition.node.on.node.as_str()),
                            ));
                        }
                    }
                    Selection::InlineFragment(inline_fragment) => {
                        let new_type_condition = inline_fragment
                            .node
                            .type_condition
                            .as_ref()
                            .map(|tc| tc.node.on.node.as_str())
                            .or(active_type_condition);
                        self.iter.push((
                            inline_fragment.node.selection_set.node.items.iter(),
                            new_type_condition,
                        ));
                    }
                },
                None => {
                    self.iter.pop();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[derive(SimpleObject)]
    #[graphql(internal)]
    struct Detail {
        x: i32,
    }

    #[derive(SimpleObject)]
    #[graphql(internal)]
    struct MyObj {
        a: i32,
        b: i32,
        c: i32,
        detail: Detail,
    }

    fn collect_pairs<'a>(
        it: impl Iterator<Item = SelectionField<'a>>,
    ) -> Vec<(String, Option<String>)> {
        it.map(|f| {
            (
                f.name().to_string(),
                f.type_condition().map(str::to_string),
            )
        })
        .collect()
    }

    struct Query;

    #[Object(internal)]
    impl Query {
        async fn obj(&self, ctx: &Context<'_>, n: i32) -> MyObj {
            let field = ctx.field();

            assert_eq!(
                field.type_condition(),
                None,
                "ctx.field().type_condition() must be None (n={})",
                n,
            );

            match n {
                1 => {
                    let pairs = collect_pairs(field.selection_set());
                    assert_eq!(
                        pairs,
                        vec![
                            ("a".into(), None),
                            ("b".into(), None),
                            ("c".into(), None),
                        ]
                    );
                }
                2 => {
                    let pairs = collect_pairs(field.selection_set());
                    assert_eq!(
                        pairs,
                        vec![
                            ("a".into(), None),
                            ("b".into(), Some("MyObj".into())),
                            ("c".into(), Some("MyObj".into())),
                        ]
                    );
                }
                3 => {
                    let pairs = collect_pairs(field.selection_set());
                    assert_eq!(
                        pairs,
                        vec![
                            ("a".into(), None),
                            ("b".into(), Some("MyObj".into())),
                            ("c".into(), Some("MyObj".into())),
                        ]
                    );
                }
                4 => {
                    let pairs = collect_pairs(field.selection_set());
                    assert_eq!(pairs, vec![("a".into(), None)]);
                }
                5 => {
                    let pairs = collect_pairs(field.selection_set());
                    assert_eq!(pairs, vec![("a".into(), Some("MyObj".into()))]);
                }
                6 => {
                    let pairs = collect_pairs(field.selection_set());
                    assert_eq!(pairs, vec![("a".into(), Some("MyObj".into()))]);
                }
                7 => {
                    let pairs = collect_pairs(field.selection_set());
                    assert_eq!(pairs, vec![("a".into(), Some("MyObj".into()))]);
                }
                8 => {
                    let detail = field
                        .selection_set()
                        .find(|f| f.name() == "detail")
                        .expect("detail not found in selection set");
                    assert_eq!(detail.type_condition(), Some("MyObj"));
                    let detail_children = collect_pairs(detail.selection_set());
                    assert_eq!(detail_children, vec![("x".into(), None)]);
                }
                9 => {
                    let look_fields = ctx.look_ahead().field("a").selection_fields();
                    assert!(!look_fields.is_empty(), "expected look_ahead to find `a`");
                    for f in &look_fields {
                        assert_eq!(f.type_condition(), Some("MyObj"));
                    }
                }
                10 => {}
                _ => panic!("unexpected n={}", n),
            }

            MyObj {
                a: 0,
                b: 0,
                c: 0,
                detail: Detail { x: 0 },
            }
        }
    }

    async fn run_ok(schema: &Schema<Query, EmptyMutation, EmptySubscription>, query: &str) {
        let resp = schema.execute(query).await;
        assert!(
            resp.errors.is_empty(),
            "execute failed: {:?} for query {}",
            resp.errors,
            query,
        );
    }

    #[tokio::test]
    async fn test_selection_field_type_condition() {
        let schema = Schema::new(Query, EmptyMutation, EmptySubscription);

        run_ok(&schema, r#"{ obj(n: 1) { a b c } }"#).await;

        run_ok(&schema, r#"{ obj(n: 2) { a ... on MyObj { b c } } }"#).await;

        run_ok(
            &schema,
            r#"{ obj(n: 3) { a ... F } } fragment F on MyObj { b c }"#,
        )
        .await;

        run_ok(&schema, r#"{ obj(n: 4) { ... { a } } }"#).await;

        run_ok(&schema, r#"{ obj(n: 5) { ... on MyObj { ... { a } } } }"#).await;

        run_ok(
            &schema,
            r#"{ obj(n: 6) { ... on MyObj { ... on MyObj { a } } } }"#,
        )
        .await;

        run_ok(
            &schema,
            r#"{ obj(n: 7) { ... on MyObj { ... F } } } fragment F on MyObj { a }"#,
        )
        .await;

        run_ok(
            &schema,
            r#"{ obj(n: 8) { ... on MyObj { detail { x } } } }"#,
        )
        .await;

        run_ok(&schema, r#"{ obj(n: 9) { ... on MyObj { a } } }"#).await;

        run_ok(&schema, r#"{ ... on Query { obj(n: 10) { a } } }"#).await;
    }
}
