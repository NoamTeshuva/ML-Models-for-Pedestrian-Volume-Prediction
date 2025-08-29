import { API_BASE } from '../config';
import type { FeatureCollection } from '../types/geo';

type Scenario = { type: 'edge_widen'; factor: number };

export async function simulate(place: string, maxFeatures = 200, scenario: Scenario = { type: 'edge_widen', factor: 1.1 }): Promise<FeatureCollection> {
  const res = await fetch(`${API_BASE}/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ place, max_features: maxFeatures, scenario }),
  });
  if (!res.ok) throw new Error(`simulate failed: ${res.status} ${res.statusText}`);
  return res.json();
}