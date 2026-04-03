use uuid::Uuid;

/// Resolve the base external request ID before API-specific prefixes such as `chatcmpl-`.
pub fn resolve_base_request_id(
    request_id_header: Option<&str>,
    request_id: Option<&str>,
) -> String {
    request_id_header
        .or(request_id)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| Uuid::new_v4().to_string())
}
