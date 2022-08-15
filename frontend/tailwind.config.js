/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {},
  },
  varients: {
    extend: {
      backgroundColor: ['active']
    }

  },
  plugins: [require('@tailwindcss/aspect-ratio')],
}
