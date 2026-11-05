import { useCallback, useEffect, useRef, useState } from 'react';
import { ApiError } from '../services/api';

interface ApiState<T> {
  data: T | null;
  error: ApiError | null;
  loading: boolean;
}

/**
 * Fetch on mount and expose an explicit refresh. The hook never substitutes placeholder data:
 * a failed request surfaces as an error state, which is the only honest option for an
 * observability dashboard.
 */
export function useApi<T>(fetcher: () => Promise<T>, deps: unknown[] = []) {
  const [state, setState] = useState<ApiState<T>>({ data: null, error: null, loading: true });
  const mounted = useRef(true);

  const run = useCallback(async () => {
    setState((previous) => ({ ...previous, loading: true, error: null }));
    try {
      const data = await fetcher();
      if (mounted.current) setState({ data, error: null, loading: false });
    } catch (error) {
      const apiError =
        error instanceof ApiError
          ? error
          : new ApiError(0, 'unknown', (error as Error)?.message ?? 'Unknown error');
      if (mounted.current) setState({ data: null, error: apiError, loading: false });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    mounted.current = true;
    void run();
    return () => {
      mounted.current = false;
    };
  }, [run]);

  return { ...state, refresh: run, setData: (data: T) => setState({ data, error: null, loading: false }) };
}

export function usePolling(callback: () => void, intervalMs: number) {
  useEffect(() => {
    const id = window.setInterval(callback, intervalMs);
    return () => window.clearInterval(id);
  }, [callback, intervalMs]);
}
