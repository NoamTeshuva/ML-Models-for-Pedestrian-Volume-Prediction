import { useEffect, useRef, useState } from 'react';
import Map from '@arcgis/core/Map';
import MapView from '@arcgis/core/views/MapView';
import GraphicsLayer from '@arcgis/core/layers/GraphicsLayer';
import Graphic from '@arcgis/core/Graphic';
import { createModelOutputLayer } from './layers/modelOutput';
import { MODEL_OUTPUT_URL } from './config';
import { simulate } from './services/api';
import type { GJFeature } from './types/geo';

export default function App() {
  const mapDiv = useRef<HTMLDivElement>(null);
  const overlayRef = useRef<GraphicsLayer | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!mapDiv.current) return;

    const map = new Map({ basemap: 'gray-vector' });

    const modelLayer = createModelOutputLayer(MODEL_OUTPUT_URL);
    const overlay = new GraphicsLayer({ title: 'What-if Overlay' });
    overlayRef.current = overlay;

    map.addMany([modelLayer, overlay]);

    const view = new MapView({
      container: mapDiv.current,
      map,
      center: [34.7818, 32.0853], // Tel Aviv
      zoom: 12,
    });

    return () => view?.destroy();
  }, []);

  async function runSim() {
    if (!overlayRef.current) return;
    setBusy(true);
    try {
      const gj = await simulate('Tel Aviv, Israel', 200, { type: 'edge_widen', factor: 1.1 });
      overlayRef.current.removeAll();

      const graphics = (gj.features ?? []).map((f: GJFeature) => {
        const g = f.geometry;
        const paths = g.type === 'LineString' ? [g.coordinates as number[][]] : (g.coordinates as number[][][]);
        const delta = f.properties?.delta ?? 0;
        const color = delta > 0 ? '#2ecc71' : delta < 0 ? '#e74c3c' : '#7f8c8d';

        return new Graphic({
          geometry: {
            type: 'polyline',
            paths,
            spatialReference: { wkid: 4326 },
          } as any,
          attributes: f.properties,
          symbol: { type: 'simple-line', color, width: 3 } as any,
          popupTemplate: {
            title: 'What-if result',
            content: 'pred_before: {pred_before}<br/>pred_after: {pred_after}<br/>delta: {delta}',
          },
        });
      });

      overlayRef.current.addMany(graphics);
    } catch (err: any) {
      console.error(err);
      alert(err?.message || 'Simulation failed');
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ height: '100vh', width: '100vw', position: 'relative' }}>
      <div ref={mapDiv} style={{ height: '100%', width: '100%' }} />
      <div className="panel">
        <button disabled={busy} onClick={runSim}>
          {busy ? 'Running…' : 'Run What-If (Tel Aviv)'}
        </button>
      </div>
    </div>
  );
}