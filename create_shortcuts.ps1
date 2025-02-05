$WshShell = New-Object -comObject WScript.Shell

# Liste des icônes à essayer
$icons = @(
    @{name="SEO Keywords Cleaner (Balai)"; file="C:\Windows\System32\shell32.dll"; index=22},  # Icône de balai
    @{name="SEO Keywords Cleaner (Filtre)"; file="C:\Windows\System32\shell32.dll"; index=123}, # Icône de filtre
    @{name="SEO Keywords Cleaner (Étoile)"; file="C:\Windows\System32\shell32.dll"; index=44},  # Icône d'étoile
    @{name="SEO Keywords Cleaner (Loupe)"; file="C:\Windows\System32\shell32.dll"; index=209}, # Icône de loupe
    @{name="SEO Keywords Cleaner (Engrenage)"; file="C:\Windows\System32\shell32.dll"; index=317}, # Icône d'engrenage
    @{name="SEO Keywords Cleaner (Document)"; file="C:\Windows\System32\shell32.dll"; index=1} # Icône de document
)

# Créer un raccourci pour chaque icône
foreach ($icon in $icons) {
    $Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\$($icon.name).lnk")
    $Shortcut.TargetPath = "C:\Users\SaaS\Desktop\SEO Keywords Cleaner\launch_app.bat"
    $Shortcut.WorkingDirectory = "C:\Users\SaaS\Desktop\SEO Keywords Cleaner"
    $Shortcut.IconLocation = "$($icon.file),$($icon.index)"
    $Shortcut.Save()
}
