# Vorlage Abschlussarbeiten

Dieses Layout kann soll für Bachelor- und Masterarbeiten verwendet werden. Die
Vorlage kann für englische und deutsche Abschlussarbeiten verwendet werden.

In `thesis_meta.tex` können persönliche Informationen für das Layout definiert
werden.

Die Datei `master.tex` heißt "master", da es sich hierbei um die Masterdatei der
Abschlussarbeit handelt. Es hat nichts mit dem angestrebten Abschluss zu tun und
sollte nicht verändert werden.

## Anforderungen

* aktuelle LaTeX-Installation (alternativ kann [unser Docker-Image verwendet
  werden](https://gitlab.cs.uni-duesseldorf.de/cn-tsn/general/templates/latex/container_registry))
* [git-lfs](https://git-lfs.github.com/) (sorgt dafür, dass in der
  [.gitattributes](.gitattributes) gelistete Dateitypen ins LFS geschoben
  werden)

## Benutzung

Eigene LaTeX-Dateien müssen an der markierten Stelle in der `master.tex` Datei
eingebunden werden.

Es wird `latexmk` verwendet, um die Arbeit zu bauen. Der Befehl dafür sieht wie
folgt aus:

    latexmk

Wenn latexmk die PDF automatisch bauen soll, wenn sich eine der eingebundenen
Dateien verändert, so sieht der Aufruf so aus:

    latexmk -pvc

Das funktioniert analog auch mit unserem LaTeX-Image:

    docker run --rm -v `pwd`:/tex gitlab.cs.uni-duesseldorf.de:5001/cn-tsn/general/templates/latex:latest latexmk

oder mit automatischem Build, wenn eine `.tex`-Datei sich ändert:

    docker run --rm -v `pwd`:/tex gitlab.cs.uni-duesseldorf.de:5001/cn-tsn/general/templates/latex:latest latexmk -pvc

Alternativ kann man die master-Datei auch automatisch von eigenen LaTeX-Editor
bauen lassen oder manuell auf der Konsole mit pdflatex und bibtex.

### Sprache ändern

Die Sprache kann in der Datei `master.tex` geändert werden. Einfach die
entsprechenden Zeilen auskommentieren / wechseln und schon ändert sich das
Layout.
