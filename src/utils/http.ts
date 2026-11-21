export class ApiResponseError extends Error {
  public status: number;
  public isHtmlResponse: boolean;

  constructor(message: string, status: number, isHtmlResponse = false) {
    super(message);
    this.name = 'ApiResponseError';
    this.status = status;
    this.isHtmlResponse = isHtmlResponse;
  }
}

/**
 * Safely fetches JSON from backend API routes, handling non-JSON / HTML error
 * responses gracefully instead of throwing raw `SyntaxError: Unexpected token '<'`.
 */
export async function fetchJson<T = any>(
  input: RequestInfo | URL,
  init?: RequestInit
): Promise<T> {
  const res = await fetch(input, init);
  const rawText = await res.text();
  const trimmed = rawText.trimStart();

  const looksLikeHtml =
    trimmed.startsWith('<!doctype') ||
    trimmed.startsWith('<!DOCTYPE') ||
    trimmed.startsWith('<html') ||
    trimmed.startsWith('<');

  if (looksLikeHtml) {
    throw new ApiResponseError(
      `API server returned an HTML page instead of JSON (HTTP ${res.status}). If the server just restarted, please wait a moment and retry.`,
      res.status,
      true
    );
  }

  let data: any;
  try {
    data = rawText ? JSON.parse(rawText) : {};
  } catch {
    throw new ApiResponseError(
      `Invalid server response format (HTTP ${res.status}). Please retry.`,
      res.status,
      false
    );
  }

  if (!res.ok) {
    const errMsg =
      (data && (data.error || data.message)) ||
      `Request failed with status ${res.status}.`;
    throw new ApiResponseError(errMsg, res.status, false);
  }

  return data as T;
}
