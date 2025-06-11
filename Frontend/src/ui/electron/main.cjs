const { app, BrowserWindow } = require('electron')
const path = require('path')

function createWindow() {
  // Get the correct path to dist-react (now at project root)
  const distPath = path.join(__dirname, '../../../dist-react')
  const indexPath = path.join(distPath, 'index.html')

  console.log('Expected build location:', distPath)
  console.log('Full index.html path:', indexPath)

  const mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      preload: path.join(__dirname, 'preload.cjs')
    }
  })

  // First try loading from expected location
  mainWindow.loadFile(indexPath)
    .catch(err => {
      console.error('Primary load failed, trying alternative path:', err)
      // Fallback to alternative path if needed
      const altPath = path.join(__dirname, '../../dist-react/index.html')
      return mainWindow.loadFile(altPath)
    })
    .then(() => console.log('App loaded successfully'))
    .catch(err => {
      console.error('All load attempts failed:', err)
      mainWindow.loadURL(`data:text/html,<h1>App Failed to Load</h1>
        <p>${err.message}</p>
        <p>Build directory expected at: ${distPath}</p>
        <p>Run "npm run build" and try again</p>`)
    })

  mainWindow.webContents.openDevTools()
}

app.whenReady().then(createWindow)

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})