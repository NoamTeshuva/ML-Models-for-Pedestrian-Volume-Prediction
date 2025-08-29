export type LineGeom = {
  type: 'LineString' | 'MultiLineString';
  coordinates: number[][] | number[][][];
};

export type FeatureProps = {
  pred_before?: number;
  pred_after?: number;
  delta?: number;
  [k: string]: any;
};

export type GJFeature = {
  type: 'Feature';
  geometry: LineGeom;
  properties: FeatureProps;
};

export type FeatureCollection = {
  type: 'FeatureCollection';
  features: GJFeature[];
};