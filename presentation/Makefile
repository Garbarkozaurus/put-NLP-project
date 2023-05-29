# Compiles the presentation
# In case of errors, check the log file: cache/presentation.log

CACHEDIR = cache
GVDIR = graphviz
LATEX = xelatex
JOBNAME = presentation
LXFLAGS = -interaction batchmode -8bit -shell-escape -output-directory "$(CACHEDIR)" -jobname $(JOBNAME)

.PHONY: directories all graphviz main clean

all: directories graphviz main

directories:
	mkdir -p -- $(CACHEDIR)

graphviz:
	$(MAKE) -C $(GVDIR)

main: main.tex
	@echo Starting build...
	$(LATEX) $(LXFLAGS) -- $^
	$(LATEX) $(LXFLAGS) -- $^
	$(LATEX) $(LXFLAGS) -- $^

	mv -- $(CACHEDIR)/$(JOBNAME).pdf .
	@echo Build successful.

clean:
	$(MAKE) -C $(GVDIR)
	$(RM) -r -- $(JOBNAME).pdf $(CACHEDIR)/
