// Registro de Chart.js y estilo oscuro compartido por todas las gráficas.
import { Chart, registerables } from 'chart.js'

Chart.register(...registerables)
Chart.defaults.color = '#b3b3b3'
Chart.defaults.borderColor = 'rgba(255,255,255,0.08)'
Chart.defaults.font.family = "'Inter','Segoe UI','Helvetica Neue',Arial,sans-serif"
Chart.defaults.plugins.legend.labels.usePointStyle = true
Chart.defaults.plugins.tooltip.backgroundColor = '#282828'
Chart.defaults.plugins.tooltip.titleColor = '#fff'
Chart.defaults.plugins.tooltip.padding = 10

export const GREEN = '#1DB954'
export const GREEN_SOFT = 'rgba(29,185,84,0.28)'
export const GRAY = '#535353'
