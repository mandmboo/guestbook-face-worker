// Protective entry point for the Railway worker.
// The worker previously defaulted to polling Supabase every 3 seconds, even when
// there was no work. Clamp the interval to a safer minimum while preserving all
// existing queue, retry and face-processing behaviour.

const configuredInterval = Number(process.env.POLL_INTERVAL_MS || 0);
const safeMinimumInterval = 15000;

process.env.POLL_INTERVAL_MS = String(
  Number.isFinite(configuredInterval) && configuredInterval > 0
    ? Math.max(configuredInterval, safeMinimumInterval)
    : safeMinimumInterval,
);

await import("./worker.js");
