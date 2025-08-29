export const API_BASE = (import.meta.env.VITE_API_BASE as string) || 'https://pedestrian-api.onrender.com';
export const MODEL_OUTPUT_URL = (import.meta.env.VITE_MODEL_OUTPUT_URL as string) || '';
if (!MODEL_OUTPUT_URL) {
  // eslint-disable-next-line no-console
  console.warn('MODEL_OUTPUT_URL is empty. Set VITE_MODEL_OUTPUT_URL in .env');
}