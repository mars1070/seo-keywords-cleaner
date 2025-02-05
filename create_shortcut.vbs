Set WshShell = CreateObject("WScript.Shell")
strDesktop = WshShell.SpecialFolders("Desktop")
strPath = WshShell.CurrentDirectory

' Créer le raccourci sur le bureau
Set oShellLink = WshShell.CreateShortcut(strDesktop & "\SEO Keywords Cleaner.lnk")
oShellLink.TargetPath = strPath & "\run_app.bat"
oShellLink.WindowStyle = 1
oShellLink.IconLocation = "%SystemRoot%\System32\SHELL32.dll,220"
oShellLink.Description = "SEO Keywords Cleaner - Nettoyage de mots-clés avec GPT"
oShellLink.WorkingDirectory = strPath
oShellLink.Save

MsgBox "Raccourci créé sur le bureau ! Double-cliquez dessus pour lancer l'application.", 64, "SEO Keywords Cleaner"
