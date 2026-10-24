// Server-only credential selection. This is configuration validation, not JWT
// verification: Supabase remains responsible for authenticating the selected key.
export function getSupabaseServerKey(env = process.env) {
  const mode = String(env.MEMORY_LANE_CREDENTIAL_MODE || "legacy-compatible").trim();
  if (!["legacy-compatible", "secret-compatible", "strict"].includes(mode)) {
    throw new Error("MEMORY_LANE_CREDENTIAL_MODE must be legacy-compatible, secret-compatible, or strict");
  }
  const secretKey = String(env.SUPABASE_SECRET_KEY || "").trim();
  const legacyKey = String(env.SUPABASE_SERVICE_ROLE_KEY || "").trim();
  const key = mode === "strict" ? secretKey
    : mode === "secret-compatible" ? secretKey || legacyKey
    : legacyKey || secretKey;

  if (!key) {
    throw new Error(mode === "strict"
      ? "SUPABASE_SECRET_KEY is required in strict credential mode"
      : "SUPABASE_SECRET_KEY or SUPABASE_SERVICE_ROLE_KEY is required");
  }
  if (key.startsWith("sb_publishable_")) {
    throw new Error("A publishable Supabase key cannot be used by a server worker");
  }
  if (key.startsWith("sb_secret_") && key.length > "sb_secret_".length) return key;
  if (mode === "strict") {
    throw new Error("SUPABASE_SECRET_KEY must be a secret key in strict credential mode");
  }

  // Reject legacy anon/user JWTs before starting privileged worker work. Never
  // include a supplied credential or decoded payload in an error or log.
  const parts = key.split(".");
  if (parts.length === 3 && parts.every((part) => /^[A-Za-z0-9_-]+$/.test(part))) {
    try {
      const payload = JSON.parse(Buffer.from(parts[1], "base64url").toString("utf8"));
      if (payload.role === "service_role") return key;
    } catch {
      // Fall through to the same non-secret configuration error.
    }
  }
  throw new Error("A Supabase secret key or legacy service_role key is required");
}
