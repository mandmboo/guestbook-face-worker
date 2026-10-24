import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";
import { getSupabaseServerKey } from "../supabase-server-key.js";

// Deliberately non-credential fixtures. No project key or service is contacted.
const secret = "sb_secret_unit_test_only";
const publicKey = "sb_publishable_unit_test_only";
function legacyJwt(role) {
  const encode = (value) => Buffer.from(JSON.stringify(value)).toString("base64url");
  return `${encode({ alg: "HS256" })}.${encode({ role })}.unit_test_signature`;
}
const legacy = legacyJwt("service_role");

test("new key remains staged until an explicit mode switch", () => {
  assert.equal(getSupabaseServerKey({ SUPABASE_SECRET_KEY: secret, SUPABASE_SERVICE_ROLE_KEY: legacy }), legacy);
  assert.equal(getSupabaseServerKey({ MEMORY_LANE_CREDENTIAL_MODE: "secret-compatible", SUPABASE_SECRET_KEY: secret, SUPABASE_SERVICE_ROLE_KEY: legacy }), secret);
  assert.equal(getSupabaseServerKey({ MEMORY_LANE_CREDENTIAL_MODE: "secret-compatible", SUPABASE_SERVICE_ROLE_KEY: legacy }), legacy);
});

test("existing legacy configuration stays compatible before cutover", () => {
  assert.equal(getSupabaseServerKey({ SUPABASE_SERVICE_ROLE_KEY: legacy }), legacy);
  assert.equal(getSupabaseServerKey({ MEMORY_LANE_CREDENTIAL_MODE: "legacy-compatible", SUPABASE_SERVICE_ROLE_KEY: legacy }), legacy);
});

test("trim whitespace and fall back only when modern key is blank", () => {
  assert.equal(getSupabaseServerKey({ SUPABASE_SECRET_KEY: ` ${secret}\n` }), secret);
  assert.equal(getSupabaseServerKey({ SUPABASE_SECRET_KEY: " \n", SUPABASE_SERVICE_ROLE_KEY: ` ${legacy} ` }), legacy);
});

test("missing and blank server keys fail without leaking values", () => {
  for (const env of [{}, { SUPABASE_SECRET_KEY: " ", SUPABASE_SERVICE_ROLE_KEY: "\n" }]) {
    assert.throws(() => getSupabaseServerKey(env), /is required/);
  }
});

test("public keys and user/anon JWTs are not privileged credentials", () => {
  for (const key of [publicKey, legacyJwt("anon"), legacyJwt("authenticated"), "sb_secret_", "malformed.unit-test.value"]) {
    assert.throws(() => getSupabaseServerKey({ MEMORY_LANE_CREDENTIAL_MODE: "secret-compatible", SUPABASE_SECRET_KEY: key, SUPABASE_SERVICE_ROLE_KEY: legacy }), (error) => {
      assert.ok(!error.message.includes(key));
      return true;
    });
  }
});

test("strict cutover requires a modern key and never falls back to legacy", () => {
  assert.equal(getSupabaseServerKey({ MEMORY_LANE_CREDENTIAL_MODE: "strict", SUPABASE_SECRET_KEY: secret, SUPABASE_SERVICE_ROLE_KEY: legacy }), secret);
  assert.throws(() => getSupabaseServerKey({ MEMORY_LANE_CREDENTIAL_MODE: "strict", SUPABASE_SERVICE_ROLE_KEY: legacy }), /SUPABASE_SECRET_KEY is required/);
  assert.throws(() => getSupabaseServerKey({ MEMORY_LANE_CREDENTIAL_MODE: "strict", SUPABASE_SECRET_KEY: " ", SUPABASE_SERVICE_ROLE_KEY: legacy }), /SUPABASE_SECRET_KEY is required/);
  assert.throws(() => getSupabaseServerKey({ MEMORY_LANE_CREDENTIAL_MODE: "strict", SUPABASE_SECRET_KEY: legacy }), /must be a secret key/);
});

test("unknown credential mode fails instead of silently allowing fallback", () => {
  assert.throws(() => getSupabaseServerKey({ MEMORY_LANE_CREDENTIAL_MODE: "strcit", SUPABASE_SECRET_KEY: secret }), /must be legacy-compatible, secret-compatible, or strict/);
});

function initialiseWorker(filename, env) {
  const source = readFileSync(new URL(`../${filename}`, import.meta.url), "utf8");
  // Exercise the actual worker's complete startup configuration, before its
  // polling loop/HTTP listener, with the external client constructor stubbed.
  const boundary = filename === "worker.js" ? "const POLL_INTERVAL_MS" : "const jobs =";
  assert.ok(source.includes(boundary));
  assert.match(source, /import \{ getSupabaseServerKey \} from "\.\/supabase-server-key\.js";/);
  const startup = source.slice(0, source.indexOf(boundary)).replace(/^import .*;\n/gm, "");
  const calls = [];
  vm.runInNewContext(startup, {
    process: { env },
    getSupabaseServerKey: () => getSupabaseServerKey(env),
    createClient: (...args) => { calls.push(args); return {}; },
    faceapi: { env: { monkeyPatch() {} } },
    Canvas: {}, Image: {}, ImageData: {},
    console: { warn() {} },
  });
  return calls;
}

for (const filename of ["worker.js", "video-render-worker.js"]) {
  test(`${filename} startup selects modern credential and disables auth persistence`, () => {
    const calls = initialiseWorker(filename, { MEMORY_LANE_CREDENTIAL_MODE: "secret-compatible", SUPABASE_URL: "https://unit-test.invalid", SUPABASE_SECRET_KEY: secret, SUPABASE_SERVICE_ROLE_KEY: legacy });
    assert.equal(calls.length, 1);
    assert.equal(calls[0][1], secret);
    assert.equal(calls[0][2].auth.persistSession, false);
    assert.equal(calls[0][2].auth.autoRefreshToken, false);
  });

  test(`${filename} startup preserves legacy-only compatibility`, () => {
    assert.equal(initialiseWorker(filename, { SUPABASE_URL: "https://unit-test.invalid", SUPABASE_SERVICE_ROLE_KEY: legacy })[0][1], legacy);
    assert.equal(initialiseWorker(filename, { SUPABASE_URL: "https://unit-test.invalid", SUPABASE_SECRET_KEY: secret, SUPABASE_SERVICE_ROLE_KEY: legacy })[0][1], legacy);
  });

  test(`${filename} fails before creating a client without a privileged key`, () => {
    for (const key of [undefined, " ", publicKey]) {
      assert.throws(() => initialiseWorker(filename, { SUPABASE_URL: "https://unit-test.invalid", SUPABASE_SECRET_KEY: key }));
    }
    assert.throws(() => initialiseWorker(filename, { SUPABASE_SECRET_KEY: secret }), /SUPABASE_URL is required/);
  });

  test(`${filename} strict startup rejects legacy-only credentials`, () => {
    assert.throws(() => initialiseWorker(filename, { MEMORY_LANE_CREDENTIAL_MODE: "strict", SUPABASE_URL: "https://unit-test.invalid", SUPABASE_SERVICE_ROLE_KEY: legacy }), /SUPABASE_SECRET_KEY is required/);
  });
}

test("render secret selection is unchanged and separate from database credentials", () => {
  const source = readFileSync(new URL("../video-render-worker.js", import.meta.url), "utf8");
  assert.ok(source.includes('const RENDER_SECRET = process.env.RENDER_SECRET || process.env.MEMORY_FILM_RENDER_SECRET || "";'));
});

test("both Docker images include the shared credential helper", () => {
  for (const filename of ["Dockerfile", "Dockerfile.video"]) {
    const source = readFileSync(new URL(`../${filename}`, import.meta.url), "utf8");
    assert.match(source, /^COPY supabase-server-key\.js \.\/supabase-server-key\.js$/m);
  }
});
