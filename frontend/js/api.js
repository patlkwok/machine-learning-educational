/** Thin wrapper around the backend JSON API. */

async function request(path, body) {
  const response = await fetch(path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: body === undefined ? undefined : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

  let payload = null;
  try {
    payload = await response.json();
  } catch {
    /* fall through to the generic error below */
  }

  if (!response.ok) {
    const detail = payload && payload.detail;
    throw new Error(
      typeof detail === 'string' ? detail : `Request failed (${response.status})`
    );
  }
  return payload;
}

export const api = {
  catalogue: () => request('/api/algorithms'),
  generate: (options) => request('/api/generate', options),
  fit: (options) => request('/api/fit', options),
};

/** Decode a base64 byte string into a Uint8Array. */
export function decodeBytes(encoded) {
  const binary = atob(encoded);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) out[i] = binary.charCodeAt(i);
  return out;
}
