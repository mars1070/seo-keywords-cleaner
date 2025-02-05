Set WshShell = CreateObject("WScript.Shell")
strPath = WshShell.CurrentDirectory

' Créer le raccourci
Set oShellLink = WshShell.CreateShortcut(strPath & "\SEO Keywords Cleaner.lnk")
oShellLink.TargetPath = "cmd.exe"
oShellLink.Arguments = "/c call venv\Scripts\activate.bat && streamlit run app.py"
oShellLink.WindowStyle = 1
oShellLink.IconLocation = strPath & "\assets\icon.ico"
oShellLink.Description = "SEO Keywords Cleaner"
oShellLink.WorkingDirectory = strPath
oShellLink.Save

' Lancer l'application
WshShell.Run "cmd /c call venv\Scripts\activate.bat && streamlit run app.py", 0, True
