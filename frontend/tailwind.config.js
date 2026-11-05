/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        void: '#04060c',
        panel: '#090e18',
        raised: '#0e1523',
        line: '#1a2436',
        ink: '#e8eefc',
        muted: '#8b9bb8',
        accent: '#3ee6d0',
        plasma: '#7c8cff',
        warn: '#f5b544',
        danger: '#ff5c7a',
        ok: '#3ddc97',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
        mono: ['JetBrains Mono', 'SFMono-Regular', 'ui-monospace', 'monospace'],
      },
      boxShadow: {
        glow: '0 0 0 1px rgba(62,230,208,0.16), 0 12px 40px -18px rgba(62,230,208,0.35)',
      },
      backgroundImage: {
        grid: 'radial-gradient(circle at 1px 1px, rgba(139,155,184,0.12) 1px, transparent 0)',
      },
    },
  },
  plugins: [],
};
