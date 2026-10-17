; Sideris Desktop 5.0.0 installer (Inno Setup 6)
; Build from the repository root:
;   pyinstaller --noconfirm Luno.spec
;   ISCC installer/windows/SiderisSetup.iss
; The app is built onedir, so the whole dist\Sideris folder is installed.
; PrivilegesRequired=lowest: per-user install in %LOCALAPPDATA%\Programs\Sideris,
; never asks for an administrator password.

#define MyAppName "Sideris AI"
#define MyAppVersion "5.0.0"
#define MyAppPublisher "Sideris AI"
#define MyAppExeName "Sideris.exe"
#define MyAppId "{{6D0C4B2E-6F1A-4A7E-9E33-5C7B0F1A2D55}"

[Setup]
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} Desktop {#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\Sideris
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\..\dist\installer
OutputBaseFilename=Sideris-Setup-5.0.0
SetupIconFile=..\..\assets\sideris.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=yes
RestartApplications=no
MinVersion=10.0
DisableWelcomePage=no
LicenseFile=..\..\LICENSE

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: checkedonce
Name: "startup"; Description: "Start {#MyAppName} when I sign in"; GroupDescription: "Startup:"; Flags: unchecked

[Files]
Source: "..\..\dist\Sideris\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\..\README.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{userprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Comment: "Private local AI assistant"
Name: "{userprograms}\{#MyAppName} command line"; Filename: "{app}\SiderisCLI.exe"; Comment: "Sideris CLI"
Name: "{userdesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userstartup}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: startup

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
