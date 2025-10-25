# Script PowerShell para descarga robusta con progreso
$url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
$output = "ml_ddd_data\ml-25m.zip"

Write-Host "Iniciando descarga robusta de MovieLens 25M..." -ForegroundColor Green
Write-Host "Destino: $output" -ForegroundColor Cyan
Write-Host "URL: $url" -ForegroundColor Cyan

# Función para mostrar progreso
$ProgressPreference = 'Continue'

try {
    # Usar Start-BitsTransfer para descargas robustas (Windows)
    if (Get-Command Start-BitsTransfer -ErrorAction SilentlyContinue) {
        Write-Host "Usando BITS (Background Intelligent Transfer Service)..." -ForegroundColor Yellow
        Start-BitsTransfer -Source $url -Destination $output -DisplayName "MovieLens 25M Download" -Priority High
        Write-Host "Descarga completada con BITS!" -ForegroundColor Green
    } else {
        # Fallback a Invoke-WebRequest
        Write-Host "Usando Invoke-WebRequest..." -ForegroundColor Yellow
        Invoke-WebRequest -Uri $url -OutFile $output -UseBasicParsing
        Write-Host "Descarga completada!" -ForegroundColor Green
    }
    
    # Verificar archivo
    $file = Get-Item $output
    $sizeMB = [Math]::Round($file.Length / 1MB, 2)
    Write-Host "Archivo descargado: $($file.Name) - $sizeMB MB" -ForegroundColor Cyan
    
    if ($file.Length -gt 200000000) {
        Write-Host "Descarga exitosa - tamaño correcto" -ForegroundColor Green
        
        # Extraer automáticamente
        Write-Host "Extrayendo archivo..." -ForegroundColor Yellow
        Expand-Archive -Path $output -DestinationPath "ml_ddd_data" -Force
        Write-Host "Extracción completada" -ForegroundColor Green
        
        # Limpiar ZIP
        Remove-Item $output
        Write-Host "Archivo ZIP limpiado" -ForegroundColor Gray
        
        Write-Host "MovieLens 25M listo para usar!" -ForegroundColor Green
    } else {
        Write-Host "Error: Archivo muy pequeño, descarga incompleta" -ForegroundColor Red
        Remove-Item $output -Force
    }
    
} catch {
    Write-Host "Error en descarga: $($_.Exception.Message)" -ForegroundColor Red
    if (Test-Path $output) {
        Remove-Item $output -Force
        Write-Host "Archivo parcial eliminado" -ForegroundColor Gray
    }
}