$WshShell = New-Object -comObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\SEO Keywords Cleaner.lnk")
$Shortcut.TargetPath = "C:\Users\SaaS\Desktop\SEO Keywords Cleaner\launch_app.bat"
$Shortcut.WorkingDirectory = "C:\Users\SaaS\Desktop\SEO Keywords Cleaner"
$Shortcut.IconLocation = "C:\Windows\System32\shell32.dll,31"  # 31 est l'index de l'icône de la corbeille
$Shortcut.Save()
