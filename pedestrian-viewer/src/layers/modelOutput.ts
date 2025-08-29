import FeatureLayer from '@arcgis/core/layers/FeatureLayer';

export function createModelOutputLayer(url: string) {
  return new FeatureLayer({
    url,
    outFields: ['*'],
    popupTemplate: {
      title: '{name}',
      content: [
        { type: 'fields', fieldInfos: [
          { fieldName: 'pred_before', label: 'Pred Before' },
          { fieldName: 'pred_after',  label: 'Pred After' },
          { fieldName: 'delta',       label: 'Δ (After - Before)' },
        ]},
      ],
    },
    renderer: {
      type: 'simple',
      symbol: { type: 'simple-line', width: 2 },
      visualVariables: [
        {
          type: 'color',
          field: 'delta',
          stops: [
            { value: -50, color: '#c0392b' },
            { value:   0, color: '#95a5a6' },
            { value:  50, color: '#27ae60' },
          ],
        },
      ],
    } as any,
  });
}