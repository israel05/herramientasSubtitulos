Get-ChildItem -File -Filter "*.mp4" | ForEach-Object {
    # Get the file name without the extension
    
    
    $nuevoNombre= $_.BaseName + ".mp3"

    
    # Execute the command
    ffmpeg -i $_ -vn -q:a 0 -map a $nuevoNombre
   
}