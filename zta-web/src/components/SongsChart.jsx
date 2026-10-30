import { Chart } from 'react-chartjs-2'
import { GREEN } from '../lib/charts.js'

const short = (s, n = 16) => (s.length > n ? `${s.slice(0, n - 1)}…` : s)

export default function SongsChart({ songs, distanceLabel = 'Distancia' }) {
  const data = {
    labels: songs.map((s) => `${s.rank}. ${short(s.track_name)}`),
    datasets: [
      { type: 'bar', label: 'Popularidad (0-100)', data: songs.map((s) => s.popularity), backgroundColor: 'rgba(29,185,84,0.75)', borderRadius: 4, yAxisID: 'y' },
      { type: 'line', label: distanceLabel, data: songs.map((s) => s.distance), borderColor: '#ffffff', backgroundColor: '#ffffff', pointRadius: 3, tension: 0.25, yAxisID: 'y1' },
    ],
  }
  const options = {
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    scales: {
      y: { min: 0, max: 100, title: { display: true, text: 'Popularidad', color: GREEN } },
      y1: { position: 'right', grid: { drawOnChartArea: false }, title: { display: true, text: distanceLabel } },
      x: { ticks: { maxRotation: 50, minRotation: 30, font: { size: 10 } } },
    },
    plugins: { legend: { position: 'bottom' } },
  }
  return <div className="chart-box"><Chart type="bar" data={data} options={options} /></div>
}
