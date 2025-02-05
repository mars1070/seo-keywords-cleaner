# Supprimer tous les raccourcis sauf celui avec le balai
Remove-Item "$env:USERPROFILE\Desktop\SEO Keywords Cleaner (Filtre).lnk" -ErrorAction SilentlyContinue
Remove-Item "$env:USERPROFILE\Desktop\SEO Keywords Cleaner (Étoile).lnk" -ErrorAction SilentlyContinue
Remove-Item "$env:USERPROFILE\Desktop\SEO Keywords Cleaner (Loupe).lnk" -ErrorAction SilentlyContinue
Remove-Item "$env:USERPROFILE\Desktop\SEO Keywords Cleaner (Engrenage).lnk" -ErrorAction SilentlyContinue
Remove-Item "$env:USERPROFILE\Desktop\SEO Keywords Cleaner (Document).lnk" -ErrorAction SilentlyContinue
Remove-Item "$env:USERPROFILE\Desktop\SEO Keywords Cleaner.lnk" -ErrorAction SilentlyContinue

# Renommer le raccourci du balai pour qu'il soit plus propre
Rename-Item "$env:USERPROFILE\Desktop\SEO Keywords Cleaner (Balai).lnk" "$env:USERPROFILE\Desktop\SEO Keywords Cleaner.lnk" -ErrorAction SilentlyContinue
