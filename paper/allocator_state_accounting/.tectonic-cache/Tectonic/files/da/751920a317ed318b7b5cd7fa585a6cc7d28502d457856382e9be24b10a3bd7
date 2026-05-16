%%
%% IEEEtran.cls 2015/08/26 version V1.8b
%% 
%% This is the IEEEtran LaTeX class for authors of the Institute of 
%% Electrical and Electronics Engineers (IEEE) Transactions journals and
%% conferences.
%% 
%% Support sites:
%% http://www.michaelshell.org/tex/ieeetran/
%% http://www.ctan.org/pkg/ieeetran
%% and
%% http://www.ieee.org/
%%
%% Based on the original 1993 IEEEtran.cls, but with many bug fixes
%% and enhancements (from both JVH and MDS) over the 1996/7 version.
%%
%%
%% Contributors:
%% Gerry Murray (1993), Silvano Balemi (1993),
%% Jon Dixon (1996), Peter N"uchter (1996),
%% Juergen von Hagen (2000), and Michael Shell (2001-2014)
%% 
%% 
%% Copyright (c) 1993-2000 by Gerry Murray, Silvano Balemi, 
%%                         Jon Dixon, Peter N"uchter,
%%                         Juergen von Hagen
%%                         and
%% Copyright (c) 2001-2015 by Michael Shell
%%
%% Current maintainer (V1.3 to V1.8b): Michael Shell
%%                                     See:
%%                                     http://www.michaelshell.org/
%%                                     for current contact information.
%%
%% Special thanks to Peter Wilson (CUA) and Donald Arseneau
%% for allowing the inclusion of the \@ifmtarg command 
%% from their ifmtarg LaTeX package. 
%% 
%%*************************************************************************
%% Legal Notice:
%% This code is offered as-is without any warranty either expressed or
%% implied; without even the implied warranty of MERCHANTABILITY or
%% FITNESS FOR A PARTICULAR PURPOSE! 
%% User assumes all risk.
%% In no event shall the IEEE or any contributor to this code be liable for
%% any damages or losses, including, but not limited to, incidental,
%% consequential, or any other damages, resulting from the use or misuse
%% of any information contained here.
%%
%% All comments are the opinions of their respective authors and are not
%% necessarily endorsed by the IEEE.
%%
%% This work is distributed under the LaTeX Project Public License (LPPL)
%% ( http://www.latex-project.org/ ) version 1.3, and may be freely used,
%% distributed and modified. A copy of the LPPL, version 1.3, is included
%% in the base LaTeX documentation of all distributions of LaTeX released
%% 2003/12/01 or later.
%% Retain all contribution notices and credits.
%% ** Modified files should be clearly indicated as such, including  **
%% ** renaming them and changing author support contact information. **
%%
%% File list of work: IEEEtran.cls, IEEEtran_HOWTO.pdf, bare_adv.tex,
%%                    bare_conf.tex, bare_jrnl.tex, bare_conf_compsoc.tex,
%%                    bare_jrnl_compsoc.tex
%% 
%% Major changes to the user interface should be indicated by an 
%% increase in the version numbers. If a version is a beta, it will 
%% be indicated with a BETA suffix, i.e., 1.4 BETA.
%% Small changes can be indicated by appending letters to the version
%% such as "IEEEtran_v14a.cls".
%% In all cases, \Providesclass, any \typeout messages to the user,
%% \IEEEtransversionmajor and \IEEEtransversionminor must reflect the
%% correct version information.
%% The changes should also be documented via source comments.
%%*************************************************************************
%%
%
% Available class options 
% e.g., \documentclass[10pt,conference]{IEEEtran} 
% 
%             *** choose only one from each category ***
%
% 9pt, 10pt, 11pt, 12pt
%    Sets normal font size. The default is 10pt.
% 
% conference, journal, technote, peerreview, peerreviewca
%    determines format mode - conference papers, journal papers,
%    correspondence papers (technotes), or peer review papers. The user
%    should also select 9pt when using technote. peerreview is like
%    journal mode, but provides for a single-column "cover" title page for
%    anonymous peer review. The paper title (without the author names) is
%    repeated at the top of the page after the cover page. For peer review
%    papers, the \IEEEpeerreviewmaketitle command must be executed (will
%    automatically be ignored for non-peerreview modes) at the place the
%    cover page is to end, usually just after the abstract (keywords are
%    not normally used with peer review papers). peerreviewca is like
%    peerreview, but allows the author names to be entered and formatted
%    as with conference mode so that author affiliation and contact
%    information can be easily seen on the cover page.
%    The default is journal.
%
% draft, draftcls, draftclsnofoot, final
%    determines if paper is formatted as a widely spaced draft (for
%    handwritten editor comments) or as a properly typeset final version.
%    draftcls restricts draft mode to the class file while all other LaTeX
%    packages (i.e., \usepackage{graphicx}) will behave as final - allows
%    for a draft paper with visible figures, etc. draftclsnofoot is like
%    draftcls, but does not display the date and the word "DRAFT" at the foot
%    of the pages. If using one of the draft modes, the user will probably
%    also want to select onecolumn.
%    The default is final.
%
% letterpaper, a4paper, cspaper
%    determines paper size: 8.5in X 11in, 210mm X 297mm or 7.875in X 10.75in.
%    Changing the paper size in the standard journal and conference modes
%    will not alter the typesetting of the document - only the margins will
%    be affected. In particular, documents using the a4paper option will
%    have reduced side margins (A4 is narrower than US letter) and a longer
%    bottom margin (A4 is longer than US letter). For both cases, the top
%    margins will be the same and the text will be horizontally centered.
%    For the compsoc conference and draft modes, it is the margins that will
%    remain constant, and thus the text area size will vary, with changes in
%    the paper size.
%    The cspaper option is the special ``trim'' paper size (7.875in x 10.75in)
%    used in the actual publication of Computer Society journals. Under
%    compsoc journal mode, this option does not alter the typesetting of the
%    document. Authors should invoke the cspaper option only if requested to
%    do so by the editors of the specific journal they are submitting to.
%    For final submission to the IEEE, authors should generally use US letter 
%    (8.5 X 11in) paper unless otherwise instructed. Note that authors should
%    ensure that all post-processing (ps, pdf, etc.) uses the same paper
%    specificiation as the .tex document. Problems here are by far the number
%    one reason for incorrect margins. IEEEtran will automatically set the
%    default paper size under pdflatex (without requiring any change to
%    pdftex.cfg), so this issue is more important to dvips users. Fix
%    config.ps, config.pdf, or ~/.dvipsrc for dvips, or use the
%    dvips -t papersize option instead as needed. For the cspaper option,
%    the corresponding dvips paper name is "ieeecs".
%    See the testflow documentation
%    http://www.ctan.org/tex-archive/macros/latex/contrib/IEEEtran/testflow
%    for more details on dvips paper size configuration.
%    The default is letterpaper.
%
% oneside, twoside
%    determines if layout follows single sided or two sided (duplex)
%    printing. The only notable change is with the headings at the top of
%    the pages.
%    The default is oneside.
%
% onecolumn, twocolumn
%    determines if text is organized into one or two columns per page. One
%    column mode is usually used only with draft papers.
%    The default is twocolumn.
%
% comsoc, compsoc, transmag
%    Use the format of the IEEE Communications Society, IEEE Computer Society
%    or IEEE Transactions on Magnetics, respectively.
%
% romanappendices
%    Use the "Appendix I" convention when numbering appendices. IEEEtran.cls
%    now defaults to Alpha "Appendix A" convention - the opposite of what
%    v1.6b and earlier did.
%
% captionsoff
%    disables the display of the figure/table captions. Some IEEE journals
%    request that captions be removed and figures/tables be put on pages
%    of their own at the end of an initial paper submission. The endfloat
%    package can be used with this class option to achieve this format.
%
% nofonttune
%    turns off tuning of the font interword spacing. Maybe useful to those
%    not using the standard Times fonts or for those who have already "tuned"
%    their fonts.
%    The default is to enable IEEEtran to tune font parameters.
%
%
%----------
% Available CLASSINPUTs provided (all are macros unless otherwise noted):
% \CLASSINPUTbaselinestretch
% \CLASSINPUTinnersidemargin
% \CLASSINPUToutersidemargin
% \CLASSINPUTtoptextmargin
% \CLASSINPUTbottomtextmargin
%
% Available CLASSINFOs provided:
% \ifCLASSINFOpdf                       (TeX if conditional)
% \CLASSINFOpaperwidth                  (macro)
% \CLASSINFOpaperheight                 (macro)
% \CLASSINFOnormalsizebaselineskip      (length)
% \CLASSINFOnormalsizeunitybaselineskip (length)
%
% Available CLASSOPTIONs provided:
% all class option flags (TeX if conditionals) unless otherwise noted,
% e.g., \ifCLASSOPTIONcaptionsoff
% point size options provided as a single macro:
% \CLASSOPTIONpt
% which will be defined as 9, 10, 11, or 12 depending on the document's
% normalsize point size.
% also, class option peerreviewca implies the use of class option peerreview
% and classoption draft implies the use of class option draftcls





\ProvidesClass{IEEEtran}[2015/08/26 V1.8b by Michael Shell]
\typeout{-- See the "IEEEtran_HOWTO" manual for usage information.}
\typeout{-- http://www.michaelshell.org/tex/ieeetran/}
\NeedsTeXFormat{LaTeX2e}

% IEEEtran.cls version numbers, provided as of V1.3
% These values serve as a way a .tex file can
% determine if the new features are provided.
% The version number of this IEEEtrans.cls can be obtained from 
% these values. i.e., V1.4
% KEEP THESE AS INTEGERS! i.e., NO {4a} or anything like that-
% (no need to enumerate "a" minor changes here)
\def\IEEEtransversionmajor{1}
\def\IEEEtransversionminor{8}


% hook to allow easy changeover to IEEEtran.cls/tools.sty error reporting
\def\@IEEEclspkgerror{\ClassError{IEEEtran}}


% These do nothing, but provide them like in article.cls
\newif\if@restonecol
\newif\if@titlepage


% class option conditionals
\newif\ifCLASSOPTIONonecolumn       \CLASSOPTIONonecolumnfalse
\newif\ifCLASSOPTIONtwocolumn       \CLASSOPTIONtwocolumntrue

\newif\ifCLASSOPTIONoneside         \CLASSOPTIONonesidetrue
\newif\ifCLASSOPTIONtwoside         \CLASSOPTIONtwosidefalse

\newif\ifCLASSOPTIONfinal           \CLASSOPTIONfinaltrue
\newif\ifCLASSOPTIONdraft           \CLASSOPTIONdraftfalse
\newif\ifCLASSOPTIONdraftcls        \CLASSOPTIONdraftclsfalse
\newif\ifCLASSOPTIONdraftclsnofoot  \CLASSOPTIONdraftclsnofootfalse

\newif\ifCLASSOPTIONpeerreview      \CLASSOPTIONpeerreviewfalse
\newif\ifCLASSOPTIONpeerreviewca    \CLASSOPTIONpeerreviewcafalse

\newif\ifCLASSOPTIONjournal         \CLASSOPTIONjournaltrue
\newif\ifCLASSOPTIONconference      \CLASSOPTIONconferencefalse
\newif\ifCLASSOPTIONtechnote        \CLASSOPTIONtechnotefalse

\newif\ifCLASSOPTIONnofonttune      \CLASSOPTIONnofonttunefalse

\newif\ifCLASSOPTIONcaptionsoff     \CLASSOPTIONcaptionsofffalse

\newif\ifCLASSOPTIONcomsoc          \CLASSOPTIONcomsocfalse
\newif\ifCLASSOPTIONcompsoc         \CLASSOPTIONcompsocfalse
\newif\ifCLASSOPTIONtransmag        \CLASSOPTIONtransmagfalse

\newif\ifCLASSOPTIONromanappendices \CLASSOPTIONromanappendicesfalse


% class info conditionals

% indicates if pdf (via pdflatex) output
\newif\ifCLASSINFOpdf               \CLASSINFOpdffalse


% V1.6b internal flag to show if using a4paper
\newif\if@IEEEusingAfourpaper       \@IEEEusingAfourpaperfalse
% V1.6b internal flag to show if using cspaper
\newif\if@IEEEusingcspaper          \@IEEEusingcspaperfalse


% IEEEtran class scratch pad registers
% dimen
\newdimen\@IEEEtrantmpdimenA
\newdimen\@IEEEtrantmpdimenB
\newdimen\@IEEEtrantmpdimenC
% count
\newcount\@IEEEtrantmpcountA
\newcount\@IEEEtrantmpcountB
\newcount\@IEEEtrantmpcountC
% token list
\newtoks\@IEEEtrantmptoksA

% we use \CLASSOPTIONpt so that we can ID the point size (even for 9pt docs)
% as well as LaTeX's \@ptsize to retain some compatability with some
% external packages
\def\@ptsize{0}
% LaTeX does not support 9pt, so we set \@ptsize to 0 - same as that of 10pt
\DeclareOption{9pt}{\def\CLASSOPTIONpt{9}\def\@ptsize{0}}
\DeclareOption{10pt}{\def\CLASSOPTIONpt{10}\def\@ptsize{0}}
\DeclareOption{11pt}{\def\CLASSOPTIONpt{11}\def\@ptsize{1}}
\DeclareOption{12pt}{\def\CLASSOPTIONpt{12}\def\@ptsize{2}}



\DeclareOption{letterpaper}{\setlength{\paperwidth}{8.5in}%
                            \setlength{\paperheight}{11in}%
                            \@IEEEusingAfourpaperfalse
                            \@IEEEusingcspaperfalse
                            \def\CLASSOPTIONpaper{letter}%
                            \def\CLASSINFOpaperwidth{8.5in}%
                            \def\CLASSINFOpaperheight{11in}}


\DeclareOption{a4paper}{\setlength{\paperwidth}{210mm}%
                        \setlength{\paperheight}{297mm}%
                        \@IEEEusingAfourpapertrue
                        \@IEEEusingcspaperfalse
                        \def\CLASSOPTIONpaper{a4}%
                        \def\CLASSINFOpaperwidth{210mm}%
                        \def\CLASSINFOpaperheight{297mm}}

% special paper option for compsoc journals
\DeclareOption{cspaper}{\setlength{\paperwidth}{7.875in}%
                        \setlength{\paperheight}{10.75in}%
                        \@IEEEusingcspapertrue
                        \@IEEEusingAfourpaperfalse
                        \def\CLASSOPTIONpaper{ieeecs}%
                        \def\CLASSINFOpaperwidth{7.875in}%
                        \def\CLASSINFOpaperheight{10.75in}}

\DeclareOption{oneside}{\@twosidefalse\@mparswitchfalse
                        \CLASSOPTIONonesidetrue\CLASSOPTIONtwosidefalse}
\DeclareOption{twoside}{\@twosidetrue\@mparswitchtrue
                        \CLASSOPTIONtwosidetrue\CLASSOPTIONonesidefalse}

\DeclareOption{onecolumn}{\CLASSOPTIONonecolumntrue\CLASSOPTIONtwocolumnfalse}
\DeclareOption{twocolumn}{\CLASSOPTIONtwocolumntrue\CLASSOPTIONonecolumnfalse}

% If the user selects draft, then this class AND any packages
% will go into draft mode.
\DeclareOption{draft}{\CLASSOPTIONdrafttrue\CLASSOPTIONdraftclstrue
                      \CLASSOPTIONdraftclsnofootfalse} 
% draftcls is for a draft mode which will not affect any packages
% used by the document.
\DeclareOption{draftcls}{\CLASSOPTIONdraftfalse\CLASSOPTIONdraftclstrue
                         \CLASSOPTIONdraftclsnofootfalse} 
% draftclsnofoot is like draftcls, but without the footer.
\DeclareOption{draftclsnofoot}{\CLASSOPTIONdraftfalse\CLASSOPTIONdraftclstrue
                               \CLASSOPTIONdraftclsnofoottrue} 
\DeclareOption{final}{\CLASSOPTIONdraftfalse\CLASSOPTIONdraftclsfalse
                      \CLASSOPTIONdraftclsnofootfalse}

\DeclareOption{journal}{\CLASSOPTIONpeerreviewfalse\CLASSOPTIONpeerreviewcafalse
                        \CLASSOPTIONjournaltrue\CLASSOPTIONconferencefalse\CLASSOPTIONtechnotefalse}

\DeclareOption{conference}{\CLASSOPTIONpeerreviewfalse\CLASSOPTIONpeerreviewcafalse
                           \CLASSOPTIONjournalfalse\CLASSOPTIONconferencetrue\CLASSOPTIONtechnotefalse}

\DeclareOption{technote}{\CLASSOPTIONpeerreviewfalse\CLASSOPTIONpeerreviewcafalse
                         \CLASSOPTIONjournalfalse\CLASSOPTIONconferencefalse\CLASSOPTIONtechnotetrue}

\DeclareOption{peerreview}{\CLASSOPTIONpeerreviewtrue\CLASSOPTIONpeerreviewcafalse
                           \CLASSOPTIONjournalfalse\CLASSOPTIONconferencefalse\CLASSOPTIONtechnotefalse}

\DeclareOption{peerreviewca}{\CLASSOPTIONpeerreviewtrue\CLASSOPTIONpeerreviewcatrue
                             \CLASSOPTIONjournalfalse\CLASSOPTIONconferencefalse\CLASSOPTIONtechnotefalse}

\DeclareOption{nofonttune}{\CLASSOPTIONnofonttunetrue}

\DeclareOption{captionsoff}{\CLASSOPTIONcaptionsofftrue}

\DeclareOption{comsoc}{\CLASSOPTIONcomsoctrue\CLASSOPTIONcompsocfalse\CLASSOPTIONtransmagfalse}

\DeclareOption{compsoc}{\CLASSOPTIONcomsocfalse\CLASSOPTIONcompsoctrue\CLASSOPTIONtransmagfalse}

\DeclareOption{transmag}{\CLASSOPTIONtransmagtrue\CLASSOPTIONcomsocfalse\CLASSOPTIONcompsocfalse}

\DeclareOption{romanappendices}{\CLASSOPTIONromanappendicestrue}


% default to US letter paper, 10pt, twocolumn, one sided, final, journal
\ExecuteOptions{letterpaper,10pt,twocolumn,oneside,final,journal}
% overrride these defaults per user requests
\ProcessOptions



%% -- Command Argument Scanning Support Functions --

% Sets the category codes for punctuation to their normal values.
% For local use with argument scanning.
\def\IEEEnormalcatcodespunct{\catcode`\!=12 \catcode`\,=12 \catcode`\:=12
\catcode`\;=12 \catcode`\`=12 \catcode`\'=12 \catcode`\"=12 \catcode`\.=12
\catcode`\/=12 \catcode`\?=12 \catcode`\*=12 \catcode`\+=12 \catcode`\-=12
\catcode`\<=12 \catcode`\>=12 \catcode`\(=12 \catcode`\)=12 \catcode`\[=12
\catcode`\]=12 \catcode`\==12 \catcode`\|=12}
% Sets the category codes for numbers to their normal values.
% For local use with argument scanning.
\def\IEEEnormalcatcodesnum{\catcode`\0=12 \catcode`\1=12 \catcode`\2=12
\catcode`\3=12 \catcode`\4=12 \catcode`\5=12 \catcode`\6=12 \catcode`\7=12
\catcode`\8=12 \catcode`\9=12}
% combined action of \IEEEnormalcatcodespunct and \IEEEnormalcatcodesnum
\def\IEEEnormalcatcodes{\IEEEnormalcatcodespunct\IEEEnormalcatcodesnum}


% usage: \@IEEEextracttoken*{}
% \@IEEEextracttoken fully expands its argument (which it then stores in
% \@IEEEextracttokenarg) via \edef and then the meaning of the first
% nonbrace (but including the empty group) token found is assigned via \let
% to \@IEEEextractedtoken as well as stored in the macro
% \@IEEEextractedtokenmacro. Tokens that would otherwise be discarded during
% the acquisition of the first are stored in \@IEEEextractedtokensdiscarded,
% however their original relative brace nesting depths are not guaranteed to
% be preserved.
% If the argument is empty, or if a first nonbrace token does not exist (or
% is an empty group), \@IEEEextractedtoken will be \relax and
% \@IEEEextractedtokenmacro and \@IEEEextractedtokensdiscarded will be empty.
%
% For example:
% \@IEEEextracttoken{{{ab}{cd}}{{ef}g}}
% results in:
%
% \@IEEEextracttokenarg          ==> a macro containing {{ab}{cd}}{{ef}g}
% \@IEEEextractedtoken           ==> the letter a
% \@IEEEextractedtokenmacro      ==> a macro containing a
% \@IEEEextractedtokensdiscarded ==> a macro containing bcd{ef}g
%
% the *-star form, \@IEEEextracttoken*, does not expand its argument
% contents during processing.
\def\@IEEEextracttoken{\@ifstar{\let\@IEEEextracttokendef=\def\@@IEEEextracttoken}{\let\@IEEEextracttokendef=\edef\@@IEEEextracttoken}}

\def\@@IEEEextracttoken#1{\@IEEEextracttokendef\@IEEEextracttokenarg{#1}\relax
\def\@IEEEextractedtokensdiscarded{}\relax % initialize to empty
% if the macro is unchanged after being acquired as a single undelimited argument
% with anything after it being stripped off as a delimited argument
% we know we have one token without any enclosing braces. loop until this is true.
\let\@IEEEextracttokencurgroup\@IEEEextracttokenarg
\loop
  % trap case of an empty argument as this would cause a problem with
  % \@@@IEEEextracttoken's first (nondelimited) argument acquisition
  \ifx\@IEEEextracttokencurgroup\@empty
    \def\@IEEEextractedtokenmacro{}\relax
  \else
    \expandafter\@@@IEEEextracttoken\@IEEEextracttokencurgroup\@IEEEgeneralsequenceDELIMITER\relax
  \fi
  \ifx\@IEEEextractedtokenmacro\@IEEEextracttokencurgroup
  \else
    \let\@IEEEextracttokencurgroup=\@IEEEextractedtokenmacro
\repeat
% we can safely do a \let= here because there should be at most one token
% the relax is needed to handle the case of no token found
\expandafter\let\expandafter\@IEEEextractedtoken\@IEEEextractedtokenmacro\relax}

\def\@@@IEEEextracttoken#1#2\@IEEEgeneralsequenceDELIMITER{\def\@IEEEextractedtokenmacro{#1}\relax
\def\@@IEEEextractedtokensdiscarded{#2}\expandafter\expandafter\expandafter\def\expandafter\expandafter\expandafter
\@IEEEextractedtokensdiscarded\expandafter\expandafter\expandafter
{\expandafter\@@IEEEextractedtokensdiscarded\@IEEEextractedtokensdiscarded}}
%%
%% -- End of Command Argument Scanning Support Functions --



% Computer Society conditional execution command
\long\def\@IEEEcompsoconly#1{\relax\ifCLASSOPTIONcompsoc\relax#1\relax\fi\relax}
% inverse
\long\def\@IEEEnotcompsoconly#1{\relax\ifCLASSOPTIONcompsoc\else\relax#1\relax\fi\relax}
% compsoc conference
\long\def\@IEEEcompsocconfonly#1{\relax\ifCLASSOPTIONcompsoc\ifCLASSOPTIONconference\relax#1\relax\fi\fi\relax}
% compsoc not conference
\long\def\@IEEEcompsocnotconfonly#1{\relax\ifCLASSOPTIONcompsoc\ifCLASSOPTIONconference\else\relax#1\relax\fi\fi\relax}


% comsoc verify that newtxmath, mtpro2, mt11p or mathtime has been loaded
\def\@IEEEcomsocverifymathfont{\typeout{-- Verifying Times compatible math font.}\relax
  \@ifpackageloaded{newtxmath}{\typeout{-- newtxmath loaded, OK.}}{\@@IEEEcomsocverifymathfont}}
\def\@@IEEEcomsocverifymathfont{\@ifpackageloaded{mtpro2}{\typeout{-- mtpro2 loaded, OK.}}{\@@@IEEEcomsocverifymathfont}}
\def\@@@IEEEcomsocverifymathfont{\@ifpackageloaded{mt11p}{\typeout{-- mt11p2 loaded, OK.}}{\@@@@IEEEcomsocverifymathfont}}
\def\@@@@IEEEcomsocverifymathfont{\@ifpackageloaded{mathtime}{\typeout{-- mathtime loaded, OK.}}{\@IEEEcomsocenforcemathfont}}

% comsoc, if a Times math font was not loaded by user, enforce it
\def\@IEEEcomsocenforcemathfont{\typeout{** Times compatible math font not found, forcing.}\relax
\IfFileExists{newtxmath.sty}{\typeout{-- Found newtxmath, loading.}\RequirePackage{newtxmath}}{\@@IEEEcomsocenforcemathfont}}
\def\@@IEEEcomsocenforcemathfont{\IfFileExists{mtpro2.sty}{\typeout{-- Found mtpro2, loading.}\RequirePackage{mtpro2}}{\@@@IEEEcomsocenforcemathfont}}
\def\@@@IEEEcomsocenforcemathfont{\IfFileExists{mt11p.sty}{\typeout{-- Found mt11p, loading.}\RequirePackage{mt11p}}{\@@@@IEEEcomsocenforcemathfont}}
\def\@@@@IEEEcomsocenforcemathfont{\IfFileExists{mathtime.sty}{\typeout{-- Found mathtime, loading.}\RequirePackage{mathtime}}{\@@@@@IEEEcomsocenforcemathfont}}
% if no acceptable Times math font package found, error with newtxmath requirement
\def\@@@@@IEEEcomsocenforcemathfont{\typeout{** No Times compatible math font package found. newtxmath is required.}\RequirePackage{newtxmath}}


\ifCLASSOPTIONcomsoc
  % ensure that if newtxmath is used, the cmintegrals option is also invoked
  \PassOptionsToPackage{cmintegrals}{newtxmath}
  % comsoc requires a Times like math font
  % ensure this requirement is satisfied at document start
  \AtBeginDocument{\@IEEEcomsocverifymathfont}
\fi



% The IEEE uses Times Roman font, so we'll default to Times.
% These three commands make up the entire times.sty package.
\renewcommand{\sfdefault}{phv}
\renewcommand{\rmdefault}{ptm}
\renewcommand{\ttdefault}{pcr}

% V1.7 compsoc nonconference papers, use Palatino/Palladio as the main text font,
% not Times Roman.
\@IEEEcompsocnotconfonly{\renewcommand{\rmdefault}{ppl}}

% enable the selected main text font
\normalfont\selectfont


\ifCLASSOPTIONcomsoc
  \typeout{-- Using IEEE Communications Society mode.}
\fi

\ifCLASSOPTIONcompsoc
  \typeout{-- Using IEEE Computer Society mode.}
\fi


% V1.7 conference notice message hook
\def\@IEEEconsolenoticeconference{\typeout{}%
\typeout{** Conference Paper **}%
\typeout{Before submitting the final camera ready copy, remember to:}%
\typeout{}%
\typeout{ 1. Manually equalize the lengths of two columns on the last page}%
\typeout{ of your paper;}%
\typeout{}%
\typeout{ 2. Ensure that any PostScript and/or PDF output post-processing}%
\typeout{ uses only Type 1 fonts and that every step in the generation}%
\typeout{ process uses the appropriate paper size.}%
\typeout{}}


% we can send console reminder messages to the user here
\AtEndDocument{\ifCLASSOPTIONconference\@IEEEconsolenoticeconference\fi}


% warn about the use of single column other than for draft mode
\ifCLASSOPTIONtwocolumn\else%
  \ifCLASSOPTIONdraftcls\else%
   \typeout{** ATTENTION: Single column mode is not typically used with IEEE publications.}%
  \fi%
\fi


% V1.7 improved paper size setting code.
% Set pdfpage and dvips paper sizes. Conditional tests are similar to that
% of ifpdf.sty. Retain within {} to ensure tested macros are never altered,
% even if only effect is to set them to \relax.
% if \pdfoutput is undefined or equal to relax, output a dvips special
{\@ifundefined{pdfoutput}{\AtBeginDvi{\special{papersize=\CLASSINFOpaperwidth,\CLASSINFOpaperheight}}}{%
% pdfoutput is defined and not equal to \relax
% check for pdfpageheight existence just in case someone sets pdfoutput
% under non-pdflatex. If exists, set them regardless of value of \pdfoutput.
\@ifundefined{pdfpageheight}{\relax}{\global\pdfpagewidth\paperwidth
\global\pdfpageheight\paperheight}%
% if using \pdfoutput=0 under pdflatex, send dvips papersize special
\ifcase\pdfoutput
\AtBeginDvi{\special{papersize=\CLASSINFOpaperwidth,\CLASSINFOpaperheight}}%
\else
% we are using pdf output, set CLASSINFOpdf flag
\global\CLASSINFOpdftrue
\fi}}

% let the user know the selected papersize
\typeout{-- Using \CLASSINFOpaperwidth\space x \CLASSINFOpaperheight\space
(\CLASSOPTIONpaper)\space paper.}

\ifCLASSINFOpdf
\typeout{-- Using PDF output.}
\else
\typeout{-- Using DVI output.}
\fi


% The idea hinted here is for LaTeX to generate markleft{} and markright{}
% automatically for you after you enter \author{}, \journal{},
% \journaldate{}, journalvol{}, \journalnum{}, etc.
% However, there may be some backward compatibility issues here as
% well as some special applications for IEEEtran.cls and special issues
% that may require the flexible \markleft{}, \markright{} and/or \markboth{}.
% We'll leave this as an open future suggestion.
%\newcommand{\journal}[1]{\def\@journal{#1}}
%\def\@journal{}



% pointsize values
% used with ifx to determine the document's normal size
\def\@IEEEptsizenine{9}
\def\@IEEEptsizeten{10}
\def\@IEEEptsizeeleven{11}
\def\@IEEEptsizetwelve{12}



% FONT DEFINITIONS (No sizexx.clo file needed) 
% V1.6 revised font sizes, displayskip values and
%      revised normalsize baselineskip to reduce underfull vbox problems
%      on the 58pc = 696pt = 9.5in text height we want
%      normalsize     #lines/column  baselineskip (aka leading)
%             9pt     63             11.0476pt (truncated down)
%            10pt     58             12pt      (exact)
%            11pt     52             13.3846pt (truncated down)
%            12pt     50             13.92pt   (exact)
%

% we need to store the nominal baselineskip for the given font size
% in case baselinestretch ever changes.
% this is a dimen, so it will not hold stretch or shrink
\newdimen\@IEEEnormalsizeunitybaselineskip
\@IEEEnormalsizeunitybaselineskip\baselineskip



%% ******* WARNING! *******
%%
%% Authors should not alter font sizes, baselineskip ("leading"),
%% margins or other spacing values in an attempt to squeeze more
%% material on each page.
%%
%% The IEEE's own typesetting software will restore the correct
%% values when re-typesetting/proofing the submitted document,
%% possibly resulting in unexpected article over length charges.
%%
%% ******* WARNING! *******


% 9pt option defaults
\ifx\CLASSOPTIONpt\@IEEEptsizenine
\typeout{-- This is a 9 point document.}
\def\normalsize{\@setfontsize{\normalsize}{9}{11.0476pt}}
\setlength{\@IEEEnormalsizeunitybaselineskip}{11.0476pt}
\normalsize
\abovedisplayskip 1.5ex plus 3pt minus 1pt
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip 0pt plus 3pt
\belowdisplayshortskip 1.5ex plus 3pt minus 1pt
\def\small{\@setfontsize{\small}{8.5}{10pt}}
\def\footnotesize{\@setfontsize{\footnotesize}{8}{9pt}}
\def\scriptsize{\@setfontsize{\scriptsize}{7}{8pt}}
\def\tiny{\@setfontsize{\tiny}{5}{6pt}}
% sublargesize is the same as large - 10pt
\def\sublargesize{\@setfontsize{\sublargesize}{10}{12pt}}
\def\large{\@setfontsize{\large}{10}{12pt}}
\def\Large{\@setfontsize{\Large}{12}{14pt}}
\def\LARGE{\@setfontsize{\LARGE}{14}{17pt}}
\def\huge{\@setfontsize{\huge}{17}{20pt}}
\def\Huge{\@setfontsize{\Huge}{20}{24pt}}
\fi
%
% 10pt option defaults
\ifx\CLASSOPTIONpt\@IEEEptsizeten
\typeout{-- This is a 10 point document.}
\def\normalsize{\@setfontsize{\normalsize}{10}{12.00pt}}
\setlength{\@IEEEnormalsizeunitybaselineskip}{12pt}
\normalsize
\abovedisplayskip 1.5ex plus 4pt minus 2pt
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip 0pt plus 4pt
\belowdisplayshortskip 1.5ex plus 4pt minus 2pt
\def\small{\@setfontsize{\small}{9}{10pt}}
\def\footnotesize{\@setfontsize{\footnotesize}{8}{9pt}}
\def\scriptsize{\@setfontsize{\scriptsize}{7}{8pt}}
\def\tiny{\@setfontsize{\tiny}{5}{6pt}}
% sublargesize is a tad smaller than large - 11pt
\def\sublargesize{\@setfontsize{\sublargesize}{11}{13.4pt}}
\def\large{\@setfontsize{\large}{12}{14pt}}
\def\Large{\@setfontsize{\Large}{14}{17pt}}
\def\LARGE{\@setfontsize{\LARGE}{17}{20pt}}
\def\huge{\@setfontsize{\huge}{20}{24pt}}
\def\Huge{\@setfontsize{\Huge}{24}{28pt}}
\fi
%
% 11pt option defaults
\ifx\CLASSOPTIONpt\@IEEEptsizeeleven
\typeout{-- This is an 11 point document.}
\def\normalsize{\@setfontsize{\normalsize}{11}{13.3846pt}}
\setlength{\@IEEEnormalsizeunitybaselineskip}{13.3846pt}
\normalsize
\abovedisplayskip 1.5ex plus 5pt minus 3pt
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip 0pt plus 5pt
\belowdisplayshortskip 1.5ex plus 5pt minus 3pt
\def\small{\@setfontsize{\small}{10}{12pt}}
\def\footnotesize{\@setfontsize{\footnotesize}{9}{10.5pt}}
\def\scriptsize{\@setfontsize{\scriptsize}{8}{9pt}}
\def\tiny{\@setfontsize{\tiny}{6}{7pt}}
% sublargesize is the same as large - 12pt
\def\sublargesize{\@setfontsize{\sublargesize}{12}{14pt}}
\def\large{\@setfontsize{\large}{12}{14pt}}
\def\Large{\@setfontsize{\Large}{14}{17pt}}
\def\LARGE{\@setfontsize{\LARGE}{17}{20pt}}
\def\huge{\@setfontsize{\huge}{20}{24pt}}
\def\Huge{\@setfontsize{\Huge}{24}{28pt}}
\fi
%
% 12pt option defaults
\ifx\CLASSOPTIONpt\@IEEEptsizetwelve
\typeout{-- This is a 12 point document.}
\def\normalsize{\@setfontsize{\normalsize}{12}{13.92pt}}
\setlength{\@IEEEnormalsizeunitybaselineskip}{13.92pt}
\normalsize
\abovedisplayskip 1.5ex plus 6pt minus 4pt
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip 0pt plus 6pt
\belowdisplayshortskip 1.5ex plus 6pt minus 4pt
\def\small{\@setfontsize{\small}{10}{12pt}}
\def\footnotesize{\@setfontsize{\footnotesize}{9}{10.5pt}}
\def\scriptsize{\@setfontsize{\scriptsize}{8}{9pt}}
\def\tiny{\@setfontsize{\tiny}{6}{7pt}}
% sublargesize is the same as large - 14pt
\def\sublargesize{\@setfontsize{\sublargesize}{14}{17pt}}
\def\large{\@setfontsize{\large}{14}{17pt}}
\def\Large{\@setfontsize{\Large}{17}{20pt}}
\def\LARGE{\@setfontsize{\LARGE}{20}{24pt}}
\def\huge{\@setfontsize{\huge}{22}{26pt}}
\def\Huge{\@setfontsize{\Huge}{24}{28pt}}
\fi



% V1.8a compsoc font sizes
% compsoc font sizes use bp "Postscript" point units (1/72in) 
% rather than the traditional pt (1/72.27)
\ifCLASSOPTIONcompsoc
% -- compsoc defaults --
% ** will override some of these values later **
% 9pt
\ifx\CLASSOPTIONpt\@IEEEptsizenine
\def\normalsize{\@setfontsize{\normalsize}{9bp}{11bp}}
\setlength{\@IEEEnormalsizeunitybaselineskip}{11bp}
\normalsize
\abovedisplayskip 1.5ex plus 3bp minus 1bp
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip 0bp plus 3bp
\belowdisplayshortskip 1.5ex plus 3bp minus 1bp
\def\small{\@setfontsize{\small}{8.5bp}{10bp}}
\def\footnotesize{\@setfontsize{\footnotesize}{8bp}{9bp}}
\def\scriptsize{\@setfontsize{\scriptsize}{7bp}{8bp}}
\def\tiny{\@setfontsize{\tiny}{5bp}{6bp}}
% sublargesize is the same as large - 10bp
\def\sublargesize{\@setfontsize{\sublargesize}{10bp}{12bp}}
\def\large{\@setfontsize{\large}{10bp}{12bp}}
\def\Large{\@setfontsize{\Large}{12bp}{14bp}}
\def\LARGE{\@setfontsize{\LARGE}{14bp}{17bp}}
\def\huge{\@setfontsize{\huge}{17bp}{20bp}}
\def\Huge{\@setfontsize{\Huge}{20bp}{24bp}}
\fi
%
% 10pt
\ifx\CLASSOPTIONpt\@IEEEptsizeten
\def\normalsize{\@setfontsize{\normalsize}{10bp}{12bp}}
\setlength{\@IEEEnormalsizeunitybaselineskip}{12bp}
\normalsize
\abovedisplayskip 1.5ex plus 4bp minus 2bp
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip 0pt plus 4bp
\belowdisplayshortskip 1.5ex plus 4bp minus 2bp
\def\small{\@setfontsize{\small}{9bp}{10bp}}
\def\footnotesize{\@setfontsize{\footnotesize}{8bp}{9bp}}
\def\scriptsize{\@setfontsize{\scriptsize}{7bp}{8bp}}
\def\tiny{\@setfontsize{\tiny}{5bp}{6bp}}
% sublargesize is a tad smaller than large - 11bp
\def\sublargesize{\@setfontsize{\sublargesize}{11bp}{13.5bp}}
\def\large{\@setfontsize{\large}{12bp}{14bp}}
\def\Large{\@setfontsize{\Large}{14bp}{17bp}}
\def\LARGE{\@setfontsize{\LARGE}{17bp}{20bp}}
\def\huge{\@setfontsize{\huge}{20bp}{24bp}}
\def\Huge{\@setfontsize{\Huge}{24bp}{28bp}}
\fi
%
% 11pt
\ifx\CLASSOPTIONpt\@IEEEptsizeeleven
\def\normalsize{\@setfontsize{\normalsize}{11bp}{13.5bp}}
\setlength{\@IEEEnormalsizeunitybaselineskip}{13.5bp}
\normalsize
\abovedisplayskip 1.5ex plus 5bp minus 3bp
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip 0pt plus 5bp
\belowdisplayshortskip 1.5ex plus 5bp minus 3bp
\def\small{\@setfontsize{\small}{10bp}{12bp}}
\def\footnotesize{\@setfontsize{\footnotesize}{9bp}{10.5bp}}
\def\scriptsize{\@setfontsize{\scriptsize}{8bp}{9bp}}
\def\tiny{\@setfontsize{\tiny}{6bp}{7bp}}
% sublargesize is the same as large - 12bp
\def\sublargesize{\@setfontsize{\sublargesize}{12bp}{14bp}}
\def\large{\@setfontsize{\large}{12bp}{14bp}}
\def\Large{\@setfontsize{\Large}{14bp}{17bp}}
\def\LARGE{\@setfontsize{\LARGE}{17bp}{20bp}}
\def\huge{\@setfontsize{\huge}{20bp}{24bp}}
\def\Huge{\@setfontsize{\Huge}{24bp}{28bp}}
\fi
%
% 12pt
\ifx\CLASSOPTIONpt\@IEEEptsizetwelve
\def\normalsize{\@setfontsize{\normalsize}{12bp}{14bp}}%
\setlength{\@IEEEnormalsizeunitybaselineskip}{14bp}%
\normalsize
\abovedisplayskip 1.5ex plus 6bp minus 4bp
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip 0pt plus 6bp
\belowdisplayshortskip 1.5ex plus 6bp minus 4bp
\def\small{\@setfontsize{\small}{10bp}{12bp}}
\def\footnotesize{\@setfontsize{\footnotesize}{9bp}{10.5bp}}
\def\scriptsize{\@setfontsize{\scriptsize}{8bp}{9bp}}
\def\tiny{\@setfontsize{\tiny}{6bp}{7bp}}
% sublargesize is the same as large - 14bp
\def\sublargesize{\@setfontsize{\sublargesize}{14bp}{17bp}}
\def\large{\@setfontsize{\large}{14bp}{17bp}}
\def\Large{\@setfontsize{\Large}{17bp}{20bp}}
\def\LARGE{\@setfontsize{\LARGE}{20bp}{24bp}}
\def\huge{\@setfontsize{\huge}{22bp}{26bp}}
\def\Huge{\@setfontsize{\Huge}{24bp}{28bp}}
\fi
%
% -- override defaults: compsoc journals use special normalsizes --
\ifCLASSOPTIONconference
% 
% compsoc conferences
% 9pt
\ifx\CLASSOPTIONpt\@IEEEptsizenine
\def\normalsize{\@setfontsize{\normalsize}{9bp}{10.8bp}}
\setlength{\@IEEEnormalsizeunitybaselineskip}{10.8bp}
\normalsize
\abovedisplayskip 1.5ex plus 3bp minus 1bp
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip 0bp plus 3bp
\belowdisplayshortskip 1.5ex plus 3bp minus 1bp
\fi
% 10pt
\ifx\CLASSOPTIONpt\@IEEEptsizeten
\def\normalsize{\@setfontsize{\normalsize}{10bp}{11.2bp}}
\setlength{\@IEEEnormalsizeunitybaselineskip}{11.2bp}
\normalsize
\abovedisplayskip 1.5ex plus 4bp minus 2bp
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip 0pt plus 4bp
\belowdisplayshortskip 1.5ex plus 4bp minus 2bp
\fi
% 11pt
\ifx\CLASSOPTIONpt\@IEEEptsizeeleven
\def\normalsize{\@setfontsize{\normalsize}{11bp}{13.2bp}}
\setlength{\@IEEEnormalsizeunitybaselineskip}{13.2bp}
\normalsize
\abovedisplayskip 1.5ex plus 5bp minus 3bp
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip 0pt plus 5bp
\belowdisplayshortskip 1.5ex plus 5bp minus 3bp
\fi
% 12pt
\ifx\CLASSOPTIONpt\@IEEEptsizetwelve
\def\normalsize{\@setfontsize{\normalsize}{12bp}{14.4bp}}
\setlength{\@IEEEnormalsizeunitybaselineskip}{14.4bp}
\normalsize
\abovedisplayskip 1.5ex plus 6bp minus 4bp
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip 0pt plus 6bp
\belowdisplayshortskip 1.5ex plus 6bp minus 4bp
\fi
% 
% compsoc nonconferences
\else
% 9pt
\ifx\CLASSOPTIONpt\@IEEEptsizenine
\def\normalsize{\@setfontsize{\normalsize}{9bp}{10.8bp}}
\setlength{\@IEEEnormalsizeunitybaselineskip}{10.8bp}
\normalsize
\abovedisplayskip 1.5ex plus 3bp minus 1bp
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip 0bp plus 3bp
\belowdisplayshortskip 1.5ex plus 3bp minus 1bp
\fi
% 10pt
\ifx\CLASSOPTIONpt\@IEEEptsizeten
% the official spec is 9.5bp with 11.4bp leading for 10pt,
% but measurements of proofs suggest upto 11.723bp leading
% here we'll use 11.54bp which gives 61 lines per column
% with the standard compsoc margins
\def\normalsize{\@setfontsize{\normalsize}{9.5bp}{11.54bp}}
\setlength{\@IEEEnormalsizeunitybaselineskip}{11.54bp}
\normalsize
\abovedisplayskip 1.5ex plus 4bp minus 2bp
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip 0pt plus 4bp
\belowdisplayshortskip 1.5ex plus 4bp minus 2bp
\fi
% 11pt
\ifx\CLASSOPTIONpt\@IEEEptsizeeleven
\def\normalsize{\@setfontsize{\normalsize}{11bp}{13.2bp}}
\setlength{\@IEEEnormalsizeunitybaselineskip}{13.2bp}
\normalsize
\abovedisplayskip 1.5ex plus 5bp minus 3bp
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip 0pt plus 5bp
\belowdisplayshortskip 1.5ex plus 5bp minus 3bp
\fi
% 12pt
\ifx\CLASSOPTIONpt\@IEEEptsizetwelve
\def\normalsize{\@setfontsize{\normalsize}{12bp}{14.4bp}}
\setlength{\@IEEEnormalsizeunitybaselineskip}{14.4bp}
\normalsize
\abovedisplayskip 1.5ex plus 6bp minus 4bp
\belowdisplayskip \abovedisplayskip
\abovedisplayshortskip 0pt plus 6bp
\belowdisplayshortskip 1.5ex plus 6bp minus 4bp
\fi
\fi\fi




% V1.6 The Computer Modern Fonts will issue a substitution warning for
% 24pt titles (24.88pt is used instead, but the default and correct
% Times font will scale exactly as needed) increase the substitution
% tolerance to turn off this warning.
% 
% V1.8a, the compsoc bp font sizes can also cause bogus font substitution
% warnings with footnote or scriptsize math and the $\bullet$ itemized
% list of \IEEEcompsocitemizethanks. So, increase this to 1.5pt or more.
\def\fontsubfuzz{1.7bp}


% warn the user in case they forget to use the 9pt option with
% technote
\ifCLASSOPTIONtechnote%
 \ifx\CLASSOPTIONpt\@IEEEptsizenine\else%
  \typeout{** ATTENTION: Technotes are normally 9pt documents.}%
 \fi%
\fi


% V1.7
% Improved \textunderscore to provide a much better fake _ when used with
% OT1 encoding. Under OT1, detect use of pcr or cmtt \ttfamily and use
% available true _ glyph for those two typewriter fonts.
\def\@IEEEstringptm{ptm} % Times Roman family
\def\@IEEEstringppl{ppl} % Palatino Roman family
\def\@IEEEstringphv{phv} % Helvetica Sans Serif family
\def\@IEEEstringpcr{pcr} % Courier typewriter family
\def\@IEEEstringcmtt{cmtt} % Computer Modern typewriter family
\DeclareTextCommandDefault{\textunderscore}{\leavevmode
\ifx\f@family\@IEEEstringpcr\string_\else
\ifx\f@family\@IEEEstringcmtt\string_\else
\ifx\f@family\@IEEEstringptm\kern 0em\vbox{\hrule\@width 0.5em\@height 0.5pt\kern -0.3ex}\else
\ifx\f@family\@IEEEstringppl\kern 0em\vbox{\hrule\@width 0.5em\@height 0.5pt\kern -0.3ex}\else
\ifx\f@family\@IEEEstringphv\kern -0.03em\vbox{\hrule\@width 0.62em\@height 0.52pt\kern -0.33ex}\kern -0.03em\else
\kern 0.09em\vbox{\hrule\@width 0.6em\@height 0.44pt\kern -0.63pt\kern -0.42ex}\kern 0.09em\fi\fi\fi\fi\fi\relax}




% set the default \baselinestretch
\def\baselinestretch{1}
\ifCLASSOPTIONdraftcls
  \def\baselinestretch{1.5}% default baselinestretch for draft modes
\fi 


% process CLASSINPUT baselinestretch
\ifx\CLASSINPUTbaselinestretch\@IEEEundefined
\else
  \edef\baselinestretch{\CLASSINPUTbaselinestretch} % user CLASSINPUT override
  \typeout{** ATTENTION: Overriding \string\baselinestretch\space to
           \baselinestretch\space via \string\CLASSINPUT.}
\fi

\small\normalsize % make \baselinestretch take affect




% store the normalsize baselineskip
\newdimen\CLASSINFOnormalsizebaselineskip
\CLASSINFOnormalsizebaselineskip=\baselineskip\relax
% and the normalsize unity (baselinestretch=1) baselineskip
% we could save a register by giving the user access to
% \@IEEEnormalsizeunitybaselineskip. However, let's protect
% its read only internal status
\newdimen\CLASSINFOnormalsizeunitybaselineskip
\CLASSINFOnormalsizeunitybaselineskip=\@IEEEnormalsizeunitybaselineskip\relax
% store the nominal value of jot
\newdimen\IEEEnormaljot
\IEEEnormaljot=0.25\baselineskip\relax

% set \jot
\jot=\IEEEnormaljot\relax




% V1.6, we are now going to fine tune the interword spacing
% The default interword glue for Times under TeX appears to use a
% nominal interword spacing of 25% (relative to the font size, i.e., 1em)
% a maximum of 40% and a minimum of 19%.
% For example, 10pt text uses an interword glue of:
% 
% 2.5pt plus 1.49998pt minus 0.59998pt
% 
% However, the IEEE allows for a more generous range which reduces the need
% for hyphenation, especially for two column text. Furthermore, the IEEE
% tends to use a little bit more nominal space between the words.
% The IEEE's interword spacing percentages appear to be:
% 35% nominal
% 23% minimum
% 50% maximum
% (They may even be using a tad more for the largest fonts such as 24pt.)
% 
% for bold text, the IEEE increases the spacing a little more:
% 37.5% nominal
% 23% minimum
% 55% maximum

% here are the interword spacing ratios we'll use
% for medium (normal weight)
\def\@IEEEinterspaceratioM{0.35}
\def\@IEEEinterspaceMINratioM{0.23}
\def\@IEEEinterspaceMAXratioM{0.50}

% for bold
\def\@IEEEinterspaceratioB{0.375}
\def\@IEEEinterspaceMINratioB{0.23}
\def\@IEEEinterspaceMAXratioB{0.55}


% compsoc nonconference papers use Palatino,
% tweak settings to better match the proofs
\ifCLASSOPTIONcompsoc
\ifCLASSOPTIONconference\else
% for medium (normal weight)
\def\@IEEEinterspaceratioM{0.28}
\def\@IEEEinterspaceMINratioM{0.21}
\def\@IEEEinterspaceMAXratioM{0.47}
% for bold
\def\@IEEEinterspaceratioB{0.305}
\def\@IEEEinterspaceMINratioB{0.21}
\def\@IEEEinterspaceMAXratioB{0.52}
\fi\fi


% command to revise the interword spacing for the current font under TeX:
% \fontdimen2 = nominal interword space
% \fontdimen3 = interword stretch
% \fontdimen4 = interword shrink
% since all changes to the \fontdimen are global, we can enclose these commands
% in braces to confine any font attribute or length changes
\def\@@@IEEEsetfontdimens#1#2#3{{%
\setlength{\@IEEEtrantmpdimenB}{\f@size pt}% grab the font size in pt, could use 1em instead.
\setlength{\@IEEEtrantmpdimenA}{#1\@IEEEtrantmpdimenB}%
\fontdimen2\font=\@IEEEtrantmpdimenA\relax
\addtolength{\@IEEEtrantmpdimenA}{-#2\@IEEEtrantmpdimenB}%
\fontdimen3\font=-\@IEEEtrantmpdimenA\relax
\setlength{\@IEEEtrantmpdimenA}{#1\@IEEEtrantmpdimenB}%
\addtolength{\@IEEEtrantmpdimenA}{-#3\@IEEEtrantmpdimenB}%
\fontdimen4\font=\@IEEEtrantmpdimenA\relax}}

% revise the interword spacing for each font weight
\def\@@IEEEsetfontdimens{{%
\mdseries
\@@@IEEEsetfontdimens{\@IEEEinterspaceratioM}{\@IEEEinterspaceMAXratioM}{\@IEEEinterspaceMINratioM}%
\bfseries
\@@@IEEEsetfontdimens{\@IEEEinterspaceratioB}{\@IEEEinterspaceMAXratioB}{\@IEEEinterspaceMINratioB}%
}}

% revise the interword spacing for each font shape
% \slshape is not often used for IEEE work and is not altered here. The \scshape caps are
% already a tad too large in the free LaTeX fonts (as compared to what the IEEE uses) so we
% won't alter these either.
\def\@IEEEsetfontdimens{{%
\normalfont
\@@IEEEsetfontdimens
\normalfont\itshape
\@@IEEEsetfontdimens
}}

% command to revise the interword spacing for each font size (and shape
% and weight). Only the \rmfamily is done here as \ttfamily uses a 
% fixed spacing and \sffamily is not used as the main text of IEEE papers.
\def\@IEEEtunefonts{{\selectfont\rmfamily
\tiny\@IEEEsetfontdimens
\scriptsize\@IEEEsetfontdimens
\footnotesize\@IEEEsetfontdimens
\small\@IEEEsetfontdimens
\normalsize\@IEEEsetfontdimens
\sublargesize\@IEEEsetfontdimens
\large\@IEEEsetfontdimens
\LARGE\@IEEEsetfontdimens
\huge\@IEEEsetfontdimens
\Huge\@IEEEsetfontdimens}}

% if the nofonttune class option is not given, revise the interword spacing
% now - in case IEEEtran makes any default length measurements, and make
% sure all the default fonts are loaded
\ifCLASSOPTIONnofonttune\else
\@IEEEtunefonts
\fi

% and again at the start of the document in case the user loaded different fonts
\AtBeginDocument{\ifCLASSOPTIONnofonttune\else\@IEEEtunefonts\fi}





% -- V1.8a page setup commands --

% The default sample text for calculating margins
% Note that IEEE publications use \scriptsize for headers and footers.
\def\IEEEdefaultsampletext{\normalfont\normalsize gT}
\def\IEEEdefaultheadersampletext{\normalfont\scriptsize T}% IEEE headers default to uppercase
\def\IEEEdefaultfootersampletext{\normalfont\scriptsize gT}



% usage: \IEEEsettextwidth{inner margin}{outer margin}
% Sets \textwidth to allow the specified inner and outer margins
% for the current \paperwidth.
\def\IEEEsettextwidth#1#2{\@IEEEtrantmpdimenA\paperwidth
\@IEEEtrantmpdimenB#1\relax
\advance\@IEEEtrantmpdimenA by -\@IEEEtrantmpdimenB
\@IEEEtrantmpdimenB#2\relax
\advance\@IEEEtrantmpdimenA by -\@IEEEtrantmpdimenB
\textwidth\@IEEEtrantmpdimenA}



% usage: \IEEEsetsidemargin{mode: i, o, c, a}{margin/offset}
% Sets \oddsidemargin and \evensidemargin to yield the specified margin
% of the given mode.
% The available modes are:
% i = inner margin
% o = outer margin
% c = centered, with the given offset
% a = adjust the margins using the given offset
% For the offsets, positive values increase the inner margin.
% \textwidth should be set properly for the given margins before calling this
% function.
\def\IEEEsetsidemargin#1#2{\@IEEEtrantmpdimenA #2\relax
\@IEEEextracttoken{#1}\relax
% check for mode errors
\ifx\@IEEEextractedtokenmacro\@empty
  \@IEEEclspkgerror{Empty mode type in \string\IEEEsetsidemargin\space (line \the\inputlineno).\MessageBreak
  Defaulting to `i'}{Valid modes for \string\IEEEsetsidemargin\space are: i, o, c and a.}\relax
  \let\@IEEEextractedtoken=i\relax
  \def\@IEEEextractedtokenmacro{i}\relax
\else
  \ifx\@IEEEextractedtokensdiscarded\@empty\else
    \typeout{** WARNING: \string\IEEEsetsidemargin\space mode specifiers after the first in `\@IEEEextracttokenarg' ignored (line \the\inputlineno).}\relax
  \fi
\fi
% handle each mode
\if\@IEEEextractedtoken a\relax
  \advance\oddsidemargin by \@IEEEtrantmpdimenA\relax
\else
\if\@IEEEextractedtoken c\relax
  \oddsidemargin\paperwidth
  \advance\oddsidemargin by -\textwidth
  \divide\oddsidemargin by 2\relax
  \advance\oddsidemargin by -1in\relax
  \advance\oddsidemargin by \@IEEEtrantmpdimenA\relax
\else
\if\@IEEEextractedtoken o\relax
  \oddsidemargin\paperwidth
  \advance\oddsidemargin by -\textwidth
  \advance\oddsidemargin by -\@IEEEtrantmpdimenA
  \advance\oddsidemargin by -1in\relax
\else
  \if\@IEEEextractedtoken i\relax
  \else
    \@IEEEclspkgerror{Unknown mode type `\@IEEEextractedtokenmacro' in \string\IEEEsetsidemargin\space (line \the\inputlineno).\MessageBreak
    Defaulting to `i'}%
    {Valid modes for \string\IEEEsetsidemargin\space are: i, o, c  and a.}%
  \fi
  \oddsidemargin\@IEEEtrantmpdimenA
  \advance\oddsidemargin by -1in\relax
\fi\fi\fi
% odd and even side margins both mean "inner" for single sided pages
\evensidemargin\oddsidemargin
% but are mirrors of each other when twosided is in effect
\if@twoside
  \evensidemargin\paperwidth
  \advance\evensidemargin by -\textwidth
  \advance\evensidemargin by -\oddsidemargin
  % have to compensate for both the builtin 1in LaTex offset
  % and the fact we already subtracted this offset from \oddsidemargin
  \advance\evensidemargin -2in\relax
\fi}



% usage: \IEEEsettextheight[sample text]{top text margin}{bottom text margin}
% Sets \textheight based on the specified top margin and bottom margin.
% Takes into consideration \paperheight, \topskip, and (by default) the
% the actual height and depth of the \IEEEdefaultsampletext text. 
\def\IEEEsettextheight{\@ifnextchar [{\@IEEEsettextheight}{\@IEEEsettextheight[\IEEEdefaultsampletext]}}
\def\@IEEEsettextheight[#1]#2#3{\textheight\paperheight\relax
  \@IEEEtrantmpdimenA #2\relax
  \advance \textheight by -\@IEEEtrantmpdimenA% subtract top margin
  \@IEEEtrantmpdimenA #3\relax
  \advance \textheight by -\@IEEEtrantmpdimenA% subtract bottom margin
  \advance \textheight by \topskip% add \topskip
  % subtract off everything above the top, and below the bottom, baselines
  \settoheight{\@IEEEtrantmpdimenA}{\begingroup #1\relax\relax\relax\endgroup}\relax
  \advance \textheight by -\@IEEEtrantmpdimenA
  \settodepth{\@IEEEtrantmpdimenA}{\begingroup #1\relax\relax\relax\endgroup}\relax
  \advance \textheight by -\@IEEEtrantmpdimenA}



\newdimen\IEEEquantizedlength
\IEEEquantizedlength 0sp\relax
\newdimen\IEEEquantizedlengthdiff
\IEEEquantizedlengthdiff 0sp\relax
\def\IEEEquantizedlengthint{0}

% usage: \IEEEquantizelength{mode: d, c, i}{base unit}{length}
% Sets the length \IEEEquantizedlength to be an integer multiple of the given
% (nonzero) base unit such that \IEEEquantizedlength approximates the given
% length.
% \IEEEquantizedlengthdiff is a length equal to the difference between the
% \IEEEquantizedlength and the given length.
% \IEEEquantizedlengthint is a macro containing the integer number of base units
% in \IEEEquantizedlength.
% i.e., \IEEEquantizedlength =  \IEEEquantizedlengthint * base unit
% The mode determines how \IEEEquantizedlength is quantized:
% d = always decrease (always round down \IEEEquantizeint)
% c = use the closest match
% i = always increase (always round up \IEEEquantizeint)
% In anycase, if the given length is already quantized,
% \IEEEquantizedlengthdiff will be set to zero.
\def\IEEEquantizelength#1#2#3{\begingroup
% work in isolation so as not to externally disturb the \@IEEEtrantmp
% variables
% load the argument values indirectly via \IEEEquantizedlengthdiff
% in case the user refers to our \@IEEEtrantmpdimenX, \IEEEquantizedlength,
% etc. in the arguments. we also will work with these as counters,
% i.e., in sp units
% A has the base unit
\IEEEquantizedlengthdiff #2\relax\relax\relax\relax
\@IEEEtrantmpcountA\IEEEquantizedlengthdiff
% B has the input length
\IEEEquantizedlengthdiff #3\relax\relax\relax\relax
\@IEEEtrantmpcountB\IEEEquantizedlengthdiff
\@IEEEtrantmpdimenA\the\@IEEEtrantmpcountA sp\relax
\@IEEEtrantmpdimenB\the\@IEEEtrantmpcountB sp\relax
% \@IEEEtrantmpcountC will have the quantized int
% \IEEEquantizedlength will have the quantized length
% \@IEEEtrantmpdimenC will have the quantized diff
% initialize them to zero as this is what will be
% exported if an error occurs
\@IEEEtrantmpcountC 0\relax
\IEEEquantizedlength 0sp\relax
\@IEEEtrantmpdimenC 0sp\relax
% extract mode
\@IEEEextracttoken{#1}\relax
% check for mode errors
\ifx\@IEEEextractedtokenmacro\@empty
  \@IEEEclspkgerror{Empty mode type in \string\IEEEquantizelength\space (line \the\inputlineno).\MessageBreak
  Defaulting to `d'}{Valid modes for \string\IEEEquantizelength\space are: d, c and i.}\relax
  \let\@IEEEextractedtoken=d\relax
  \def\@IEEEextractedtokenmacro{d}\relax
\else
  \ifx\@IEEEextractedtokensdiscarded\@empty\else
    \typeout{** WARNING: \string\IEEEquantizelength\space mode specifiers after the first in `\@IEEEextracttokenarg' ignored (line \the\inputlineno).}\relax
  \fi
\fi
% check for base unit is zero error
\ifnum\@IEEEtrantmpcountA=0\relax
\@IEEEclspkgerror{Base unit is zero in \string\IEEEquantizelength\space (line \the\inputlineno).\MessageBreak
  \string\IEEEquantizedlength\space and \string\IEEEquantizedlengthdiff\space are set to zero}{Division by zero is not allowed.}\relax
\else% base unit is nonzero
  % \@IEEEtrantmpcountC carries the number of integer units
  % in the quantized length (integer length \ base)
  \@IEEEtrantmpcountC\@IEEEtrantmpcountB\relax
  \divide\@IEEEtrantmpcountC by \@IEEEtrantmpcountA\relax
  % \IEEEquantizedlength has the (rounded down) quantized length
  % = base * int
  \IEEEquantizedlength\@IEEEtrantmpdimenA\relax
  \multiply\IEEEquantizedlength by \@IEEEtrantmpcountC\relax
  % \@IEEEtrantmpdimenC has the difference
  % = quantized length - length
  \@IEEEtrantmpdimenC\IEEEquantizedlength\relax
  \advance\@IEEEtrantmpdimenC by -\@IEEEtrantmpdimenB\relax
  % trap special case of length being already quantized
  % to avoid a roundup under i option
  \ifdim\@IEEEtrantmpdimenC=0sp\relax
  \else % length not is already quantized
    % set dimenA to carry the upper quantized (absolute value) difference:
    % quantizedlength + base - length
    \advance\@IEEEtrantmpdimenA by \IEEEquantizedlength\relax
    \advance\@IEEEtrantmpdimenA by -\@IEEEtrantmpdimenB\relax
    % set dimenB to carry the lower quantized (absolute value) difference:
    % length - quantizedlength
    \advance\@IEEEtrantmpdimenB by -\IEEEquantizedlength\relax
    % handle each mode
    \if\@IEEEextractedtoken c\relax
      % compare upper and lower amounts, select upper if lower > upper
      \ifdim\@IEEEtrantmpdimenB>\@IEEEtrantmpdimenA\relax
        % use upper
        \advance\IEEEquantizedlength by \the\@IEEEtrantmpcountA sp\relax
        \advance\@IEEEtrantmpcountC by 1\relax
        \@IEEEtrantmpdimenC\@IEEEtrantmpdimenA
      \else% <=. uselower
        % no need to do anything for lower, use output values already setup
      \fi
    \else% not mode c
    \if\@IEEEextractedtoken i\relax
      % always round up under i mode
      \advance\IEEEquantizedlength by \the\@IEEEtrantmpcountA sp\relax
      \advance\@IEEEtrantmpcountC by 1\relax
      \@IEEEtrantmpdimenC\@IEEEtrantmpdimenA
    \else
      \if\@IEEEextractedtoken d\relax
      \else
        \@IEEEclspkgerror{Unknown mode type `\@IEEEextractedtokenmacro' in \string\IEEEquantizelength\space (line \the\inputlineno).\MessageBreak
        Defaulting to `d'}%
        {Valid modes for \string\IEEEquantizelength\space are: d, c, and i.}\relax
      \fi % if d
      % no need to do anything for d, use output values already setup
    \fi\fi % if i, c
  \fi % if length is already quantized
\fi% if base unit is zero
% globally assign the results to macros we use here to escape the enclosing
% group without needing to call \global on any of the \@IEEEtrantmp variables.
% \@IEEEtrantmpcountC has the quantized int
% \IEEEquantizedlength has the quantized length
% \@IEEEtrantmpdimenC has the quantized diff
\xdef\@IEEEquantizedlengthintmacro{\the\@IEEEtrantmpcountC}\relax
\@IEEEtrantmpcountC\IEEEquantizedlength\relax
\xdef\@IEEEquantizedlengthmacro{\the\@IEEEtrantmpcountC}\relax
\@IEEEtrantmpcountC\@IEEEtrantmpdimenC\relax
\xdef\@IEEEquantizedlengthdiffmacro{\the\@IEEEtrantmpcountC}\relax
\endgroup
% locally assign the outputs here from the macros
\expandafter\IEEEquantizedlength\@IEEEquantizedlengthmacro sp\relax
\expandafter\IEEEquantizedlengthdiff\@IEEEquantizedlengthdiffmacro sp\relax
\edef\IEEEquantizedlengthint{\@IEEEquantizedlengthintmacro}\relax}



\newdimen\IEEEquantizedtextheightdiff
\IEEEquantizedtextheightdiff 0sp\relax

% usage: \IEEEquantizetextheight[base unit]{mode: d, c, i}
% Sets \textheight to be an integer multiple of the current \baselineskip
% (or the optionally specified base unit) plus the first (\topskip) line.
% \IEEEquantizedtextheightdiff is a length equal to the difference between
% the new quantized and original \textheight.
% \IEEEquantizedtextheightlpc is a macro containing the integer number of
% lines per column under the quantized \textheight. i.e.,
% \textheight = \IEEEquantizedtextheightlpc * \baselineskip + \topskip
% The mode determines how \textheight is quantized:
% d = always decrease (always round down the number of lines per column)
% c = use the closest match
% i = always increase (always round up the number of lines per column)
% In anycase, if \textheight is already quantized, it will remain unchanged,
% and \IEEEquantizedtextheightdiff will be set to zero.
% Depends on: \IEEEquantizelength
\def\IEEEquantizetextheight{\@ifnextchar [{\@IEEEquantizetextheight}{\@IEEEquantizetextheight[\baselineskip]}}
\def\@IEEEquantizetextheight[#1]#2{\begingroup
% use our \IEEEquantizedtextheightdiff as a scratch pad
% we need to subtract off \topskip before quantization
\IEEEquantizedtextheightdiff\textheight
\advance\IEEEquantizedtextheightdiff by -\topskip\relax
\IEEEquantizelength{#2}{#1}{\IEEEquantizedtextheightdiff}
% add back \topskip line
\advance\IEEEquantizedlength by \topskip
\@IEEEtrantmpcountC\IEEEquantizedlengthint\relax
\advance\@IEEEtrantmpcountC by 1\relax
% globally assign the results to macros we use here to escape the enclosing
% group without needing to call \global on any of the \@IEEEtrantmp variables.
\xdef\@IEEEquantizedtextheightlpcmacro{\the\@IEEEtrantmpcountC}\relax
\@IEEEtrantmpcountC\IEEEquantizedlength\relax
\xdef\@IEEEquantizedtextheightmacro{\the\@IEEEtrantmpcountC}\relax
\@IEEEtrantmpcountC\IEEEquantizedlengthdiff\relax
\xdef\@IEEEquantizedtextheightdiffmacro{\the\@IEEEtrantmpcountC}\relax
\endgroup
% locally assign the outputs here from the macros
\textheight\@IEEEquantizedtextheightmacro sp\relax
\IEEEquantizedtextheightdiff\@IEEEquantizedtextheightdiffmacro sp\relax
\edef\IEEEquantizedtextheightlpc{\@IEEEquantizedtextheightlpcmacro}}



% usage: \IEEEsettopmargin[sample text]{mode: t, b, c, a, q}{margin/offset}
% Sets \topmargin based on the specified vertical margin.
% Takes into consideration the base 1in offset, \headheight, \headsep,
% \topskip, and (by default) the the actual height (or, for the bottom, depth)
% of the \IEEEdefaultsampletext text.
% The available modes are:
% t = top margin
% b = bottom margin
% c = vertically centered, with the given offset
% a = adjust the vertical margins using the given offset
% q = adjust the margins using \IEEEquantizedtextheightdiff and the given offset
% For the offsets, positive values increase the top margin.
% \headheight, \headsep, \topskip and \textheight should be set properly for the
% given margins before calling this function.
\def\IEEEsettopmargin{\@ifnextchar [{\@IEEEsettopmargin}{\@IEEEsettopmargin[\IEEEdefaultsampletext]}}
\def\@IEEEsettopmargin[#1]#2#3{\@IEEEtrantmpdimenA #3\relax
\@IEEEextracttoken{#2}\relax
% check for mode errors
\ifx\@IEEEextractedtokenmacro\@empty
  \@IEEEclspkgerror{Empty mode type in \string\IEEEsettopmargin\space (line \the\inputlineno).\MessageBreak
  Defaulting to `t'}{Valid modes for \string\IEEEsettopmargin\space are: t, b, c, a and q.}\relax
  \let\@IEEEextractedtoken=t\relax
  \def\@IEEEextractedtokenmacro{t}\relax
\else
  \ifx\@IEEEextractedtokensdiscarded\@empty\else
    \typeout{** WARNING: \string\IEEEsettopmargin\space mode specifiers after the first in `\@IEEEextracttokenarg' ignored (line \the\inputlineno).}\relax
  \fi
\fi
% handle each mode
\if\@IEEEextractedtoken a\relax
  \advance\topmargin by \@IEEEtrantmpdimenA\relax
\else
\if\@IEEEextractedtoken q\relax
  % we need to adjust by half the \IEEEquantizedtextheightdiff value
  \@IEEEtrantmpdimenB\IEEEquantizedtextheightdiff\relax
  \divide\@IEEEtrantmpdimenB by 2\relax
  % a positive \IEEEquantizedtextheightdiff means we need to reduce \topmargin
  % because \textheight has been lenghtened
  \advance\topmargin by -\@IEEEtrantmpdimenB\relax
  \advance\topmargin by \@IEEEtrantmpdimenA\relax
\else
\if\@IEEEextractedtoken c\relax
  \topmargin\paperheight
  \advance\topmargin by -\textheight
  % \textheight includes \topskip, but we should not count topskip whitespace here, backout
  \advance \topmargin by \topskip
  \settoheight{\@IEEEtrantmpdimenB}{\begingroup #1\relax\relax\relax\endgroup}\relax
  \advance\topmargin by -\@IEEEtrantmpdimenB\relax
  \settodepth{\@IEEEtrantmpdimenB}{\begingroup #1\relax\relax\relax\endgroup}\relax
  \advance\topmargin by -\@IEEEtrantmpdimenB\relax
  \divide\topmargin by 2\relax
  \advance\topmargin by \@IEEEtrantmpdimenA\relax
\else
\if\@IEEEextractedtoken b\relax
  \topmargin\paperheight
  \advance\topmargin by -\textheight
  % \textheight includes \topskip, but we should not count topskip whitespace here, backout
  \advance \topmargin by \topskip
  \settodepth{\@IEEEtrantmpdimenB}{\begingroup #1\relax\relax\relax\endgroup}\relax
  \advance\topmargin by -\@IEEEtrantmpdimenB\relax
  \advance\topmargin by -\@IEEEtrantmpdimenA\relax
\else
  \if\@IEEEextractedtoken t\relax
  \else
    \@IEEEclspkgerror{Unknown mode type `\@IEEEextractedtokenmacro' in \string\IEEEsettopmargin\space (line \the\inputlineno).\MessageBreak
    Defaulting to `t'}%
    {Valid modes for \string\IEEEsettopmargin\space are: t, b, c, a and q.}\relax
  \fi
  \topmargin\@IEEEtrantmpdimenA\relax
  \settoheight{\@IEEEtrantmpdimenB}{\begingroup #1\relax\relax\relax\endgroup}\relax
  \advance\topmargin by \@IEEEtrantmpdimenB\relax
\fi\fi % if t, b, c
% convert desired top margin into actual \topmargin
% this is not done for the q or a modes because they are only adjustments
\advance \topmargin by -\topskip
\advance \topmargin by -1in
\advance \topmargin by -\headheight
\advance \topmargin by -\headsep
\fi\fi % if q, a
}



% usage: \IEEEsetheadermargin[header sample][text sample]{mode: t, b, c, a}{margin/offset}
% Differentially adjusts \topmargin and \headsep (such that their sum is unchanged)
% based on the specified header margin.
% Takes into consideration the base 1in offset, \headheight, \topskip, and (by default)
% the actual height (or depth) of the \IEEEdefaultheadersampletext and
% \IEEEdefaultsampletext text.
% The available modes are:
% t = top margin (top of the header text to the top of the page)
% b = bottom margin (bottom of the header text to the top of the main text)
% c = vertically centered between the main text and the top of the page,
%     with the given offset
% a = adjust the vertical position using the given offset
% For the offsets, positive values move the header downward.
% \headheight, \headsep, \topskip and \topmargin should be set properly before
% calling this function.
\def\IEEEsetheadermargin{\@ifnextchar [{\@IEEEsetheadermargin}{\@IEEEsetheadermargin[\IEEEdefaultheadersampletext]}}
\def\@IEEEsetheadermargin[#1]{\@ifnextchar [{\@@IEEEsetheadermargin[#1]}{\@@IEEEsetheadermargin[#1][\IEEEdefaultsampletext]}}
\def\@@IEEEsetheadermargin[#1][#2]#3#4{\@IEEEtrantmpdimenA #4\relax
\@IEEEextracttoken{#3}\relax
% check for mode errors
\ifx\@IEEEextractedtokenmacro\@empty
  \@IEEEclspkgerror{Empty mode type in \string\IEEEsetheadermargin\space (line \the\inputlineno).\MessageBreak
  Defaulting to `t'}{Valid modes for \string\IEEEsetheadermargin\space are: t, b, c, and a.}\relax
  \let\@IEEEextractedtoken=t\relax
  \def\@IEEEextractedtokenmacro{t}\relax
\else
  \ifx\@IEEEextractedtokensdiscarded\@empty\else
    \typeout{** WARNING: \string\IEEEsetheadermargin\space mode specifiers after the first in `\@IEEEextracttokenarg' ignored (line \the\inputlineno).}\relax
  \fi
\fi
% handle each mode
\if\@IEEEextractedtoken a\relax
  % No need to do anything here and can pass through the adjustment
  % value as is. The end adjustment of \topmargin and \headsep will
  % do all that is needed
\else
\if\@IEEEextractedtoken c\relax
  % get the bottom margin
  \@IEEEtrantmpdimenB\headsep\relax
  \settodepth{\@IEEEtrantmpdimenC}{\begingroup #1\relax\relax\relax\endgroup}\relax
  \advance\@IEEEtrantmpdimenB by -\@IEEEtrantmpdimenC
  \advance\@IEEEtrantmpdimenB by \topskip
  \settoheight{\@IEEEtrantmpdimenC}{\begingroup #2\relax\relax\relax\endgroup}\relax
  \advance\@IEEEtrantmpdimenB by -\@IEEEtrantmpdimenC
  % at this point \@IEEEtrantmpdimenB has the actual header bottom margin
  % subtract from it the top header margin
  \advance\@IEEEtrantmpdimenB -1in\relax % take into consideration the system 1in offset of the top margin
  \advance\@IEEEtrantmpdimenB by -\topmargin
  \advance\@IEEEtrantmpdimenB by -\headheight
  \settoheight{\@IEEEtrantmpdimenC}{\begingroup #1\relax\relax\relax\endgroup}\relax
  \advance\@IEEEtrantmpdimenB by \@IEEEtrantmpdimenC
  % at this point \@IEEEtrantmpdimenB has the difference between the bottom and top margins
  % we need to adjust by half this amount to center the header
  \divide\@IEEEtrantmpdimenB by 2\relax
  % and add to offset
  \advance\@IEEEtrantmpdimenA by \@IEEEtrantmpdimenB
\else
\if\@IEEEextractedtoken b\relax
  \@IEEEtrantmpdimenB\headsep\relax
  \settodepth{\@IEEEtrantmpdimenC}{\begingroup #1\relax\relax\relax\endgroup}\relax
  \advance\@IEEEtrantmpdimenB by -\@IEEEtrantmpdimenC
  \advance\@IEEEtrantmpdimenB by \topskip
  \settoheight{\@IEEEtrantmpdimenC}{\begingroup #2\relax\relax\relax\endgroup}\relax
  \advance\@IEEEtrantmpdimenB by -\@IEEEtrantmpdimenC
  % at this point \@IEEEtrantmpdimenB has the actual header bottom margin
  % get the difference between the actual and the desired
  \advance\@IEEEtrantmpdimenB by -\@IEEEtrantmpdimenA
  \@IEEEtrantmpdimenA\@IEEEtrantmpdimenB
\else
  \if\@IEEEextractedtoken t\relax
  \else
    \@IEEEclspkgerror{Unknown mode type `\@IEEEextractedtokenmacro' in \string\IEEEsetheadermargin\space (line \the\inputlineno).\MessageBreak
    Defaulting to `t'}%
    {Valid modes for \string\IEEEsetheadermargin\space are: t, b, c and a.}\relax
  \fi
  \@IEEEtrantmpdimenB 1in\relax % take into consideration the system 1in offset of the top margin
  \advance\@IEEEtrantmpdimenB by \topmargin
  \advance\@IEEEtrantmpdimenB by \headheight
  \settoheight{\@IEEEtrantmpdimenC}{\begingroup #1\relax\relax\relax\endgroup}\relax
  \advance\@IEEEtrantmpdimenB by -\@IEEEtrantmpdimenC
  % at this point \@IEEEtrantmpdimenB has the actual header top margin
  % get the difference between the desired and the actual
  \advance\@IEEEtrantmpdimenA by -\@IEEEtrantmpdimenB
\fi\fi % if t, b, c
\fi % if a
% advance \topmargin by the needed amount and reduce \headsep by the same
% so as not to disturb the location of the main text
\advance\topmargin by \@IEEEtrantmpdimenA\relax
\advance\headsep by -\@IEEEtrantmpdimenA\relax
}



% usage: \IEEEsetfootermargin[footer sample][text sample]{mode: t, b, c, a}{margin/offset}
% Adjusts \footskip based on the specified footer margin.
% Takes into consideration the base 1in offset, \paperheight, \headheight,
% \headsep, \textheight and (by default) the actual height (or depth) of the
% \IEEEdefaultfootersampletext and \IEEEdefaultsampletext text.
% The available modes are:
% t = top margin (top of the footer text to the bottom of the main text)
% b = bottom margin (bottom of the footer text to the bottom of page)
% c = vertically centered between the main text and the bottom of the page,
%     with the given offset
% a = adjust the vertical position using the given offset
% For the offsets, positive values move the footer downward.
% \headheight, \headsep, \topskip, \topmargin, and \textheight should be set
% properly before calling this function.
\def\IEEEsetfootermargin{\@ifnextchar [{\@IEEEsetfootermargin}{\@IEEEsetfootermargin[\IEEEdefaultfootersampletext]}}
\def\@IEEEsetfootermargin[#1]{\@ifnextchar [{\@@IEEEsetfootermargin[#1]}{\@@IEEEsetfootermargin[#1][\IEEEdefaultsampletext]}}
\def\@@IEEEsetfootermargin[#1][#2]#3#4{\@IEEEtrantmpdimenA #4\relax
\@IEEEextracttoken{#3}\relax
% check for mode errors
\ifx\@IEEEextractedtokenmacro\@empty
  \@IEEEclspkgerror{Empty mode type in \string\IEEEsetfootermargin\space (line \the\inputlineno).\MessageBreak
  Defaulting to `t'}{Valid modes for \string\IEEEsetfootermargin\space are: t, b, c, and a.}\relax
  \let\@IEEEextractedtoken=t\relax
  \def\@IEEEextractedtokenmacro{t}\relax
\else
  \ifx\@IEEEextractedtokensdiscarded\@empty\else
    \typeout{** WARNING: \string\IEEEsetfootermargin\space mode specifiers after the first in `\@IEEEextracttokenarg' ignored (line \the\inputlineno).}\relax
  \fi
\fi
% handle each mode
\if\@IEEEextractedtoken a\relax
  % No need to do anything here and can pass through the adjustment
  % value as is. The end adjustment of \footskip will do all that
  % is needed
\else
\if\@IEEEextractedtoken c\relax
  % calculate the bottom margin
  \@IEEEtrantmpdimenB 1in\relax % system 1in offset
  \advance\@IEEEtrantmpdimenB\topmargin\relax
  \advance\@IEEEtrantmpdimenB\headheight\relax
  \advance\@IEEEtrantmpdimenB\headsep\relax
  \advance\@IEEEtrantmpdimenB\textheight\relax
  \advance\@IEEEtrantmpdimenB\footskip\relax
  \settodepth{\@IEEEtrantmpdimenC}{\begingroup #1\relax\relax\relax\endgroup}\relax
  \advance\@IEEEtrantmpdimenC by \@IEEEtrantmpdimenB
  \@IEEEtrantmpdimenB\paperheight
  \advance\@IEEEtrantmpdimenB by -\@IEEEtrantmpdimenC
  % at this point \@IEEEtrantmpdimenB has the actual footer bottom margin
  % now subtract off the footer top margin
  \advance\@IEEEtrantmpdimenB -\footskip\relax
  \settodepth{\@IEEEtrantmpdimenC}{\begingroup #2\relax\relax\relax\endgroup}\relax
  \advance\@IEEEtrantmpdimenB by \@IEEEtrantmpdimenC
  \settoheight{\@IEEEtrantmpdimenC}{\begingroup #1\relax\relax\relax\endgroup}\relax
  \advance\@IEEEtrantmpdimenB by \@IEEEtrantmpdimenC
  % at this point \@IEEEtrantmpdimenB has the difference between the bottom
  % and top footer margins
  % our adjustment must be half this value to center the footer
  \divide\@IEEEtrantmpdimenB by 2\relax
  % add to the offset
  \advance\@IEEEtrantmpdimenA by \@IEEEtrantmpdimenB
\else
\if\@IEEEextractedtoken b\relax
  % calculate the bottom margin
  \@IEEEtrantmpdimenB 1in\relax % system 1in offset
  \advance\@IEEEtrantmpdimenB\topmargin\relax
  \advance\@IEEEtrantmpdimenB\headheight\relax
  \advance\@IEEEtrantmpdimenB\headsep\relax
  \advance\@IEEEtrantmpdimenB\textheight\relax
  \advance\@IEEEtrantmpdimenB\footskip\relax
  \settodepth{\@IEEEtrantmpdimenC}{\begingroup #1\relax\relax\relax\endgroup}\relax
  \advance\@IEEEtrantmpdimenC by \@IEEEtrantmpdimenB
  \@IEEEtrantmpdimenB\paperheight
  \advance\@IEEEtrantmpdimenB by -\@IEEEtrantmpdimenC
  % at this point \@IEEEtrantmpdimenB has the actual footer bottom margin
  % get the difference between the actual and the desired
  \advance\@IEEEtrantmpdimenB by -\@IEEEtrantmpdimenA
  \@IEEEtrantmpdimenA\@IEEEtrantmpdimenB
\else
  \if\@IEEEextractedtoken t\relax
  \else
    \@IEEEclspkgerror{Unknown mode type `\@IEEEextractedtokenmacro' in \string\IEEEsetfootermargin\space (line \the\inputlineno).\MessageBreak
    Defaulting to `t'}%
    {Valid modes for \string\IEEEsetfootermargin\space are: t, b, c and a.}\relax
  \fi
  \@IEEEtrantmpdimenB\footskip\relax
  \settodepth{\@IEEEtrantmpdimenC}{\begingroup #2\relax\relax\relax\endgroup}\relax
  \advance\@IEEEtrantmpdimenB by -\@IEEEtrantmpdimenC
  \settoheight{\@IEEEtrantmpdimenC}{\begingroup #1\relax\relax\relax\endgroup}\relax
  \advance\@IEEEtrantmpdimenB by -\@IEEEtrantmpdimenC
  % at this point \@IEEEtrantmpdimenB has the actual footer top margin
  % get the difference between the desired and the actual
  \advance\@IEEEtrantmpdimenA by -\@IEEEtrantmpdimenB
\fi\fi % if t, b, c
\fi % if a
% advance \footskip by the needed amount
\advance\footskip by \@IEEEtrantmpdimenA\relax
}

% -- End V1.8a page setup commands --





% V1.6
% LaTeX is a little to quick to use hyphenations
% So, we increase the penalty for their use and raise
% the badness level that triggers an underfull hbox
% warning. The author may still have to tweak things,
% but the appearance will be much better "right out
% of the box" than that under V1.5 and prior.
% TeX default is 50
\hyphenpenalty=750
\ifCLASSOPTIONcompsoc
\hyphenpenalty 500
\fi
% If we didn't adjust the interword spacing, 2200 might be better.
% The TeX default is 1000
\hbadness=1350
% The IEEE does not use extra spacing after punctuation
\frenchspacing

% V1.7 increase this a tad to discourage equation breaks
\binoppenalty=1000 % default 700
\relpenalty=800     % default 500

% v1.8a increase these to discourage widows and orphans
\clubpenalty=1000          % default 150
\widowpenalty=1000         % default 150
\displaywidowpenalty=1000 % default  50


% margin note stuff
\marginparsep      10pt
\marginparwidth    20pt
\marginparpush     25pt


% if things get too close, go ahead and let them touch
\lineskip            0pt
\normallineskip      0pt
\lineskiplimit       0pt
\normallineskiplimit 0pt

% The distance from the lower edge of the text body to the
% footline
\footskip 0.4in

% normally zero, should be relative to font height.
% put in a little rubber to help stop some bad breaks (underfull vboxes)
\parskip 0ex plus 0.2ex minus 0.1ex

\parindent    1.0em
\ifCLASSOPTIONcompsoc
  \parindent    1.5em
\fi

\headheight   12pt
\headsep      18pt
% use the normal font baselineskip
% so that \topskip is unaffected by changes in \baselinestretch
\topskip=\@IEEEnormalsizeunitybaselineskip


% V1.8 \maxdepth defaults to 4pt, but should be font size dependent
\maxdepth=0.5\@IEEEnormalsizeunitybaselineskip
\textheight       58pc  % 9.63in, 696pt

% set the default top margin to 58pt
% which results in a \topmargin of -49.59pt for 10pt documents
\IEEEsettopmargin{t}{58pt}
% tweak textheight to a perfect integer number of lines/column.
% standard is: 9pt/63 lpc; 10pt/58 lpc; 11pt/52 lpc; 12pt/50 lpc
\IEEEquantizetextheight{c}
% tweak top margin so that the error is shared equally at the top and bottom
\IEEEsettopmargin{q}{0sp}


\columnsep         1pc
\textwidth        43pc   % 2 x 21pc + 1pc = 43pc

% set the default side margins to center the text
\IEEEsetsidemargin{c}{0pt}


% adjust margins for default conference mode
\ifCLASSOPTIONconference
 \textheight        9.25in % The standard for conferences (668.4975pt)
 \IEEEsettopmargin{t}{0.75in}
 % tweak textheight to a perfect integer number of lines/page.
 % standard is: 9pt/61 lpc; 10pt/56 lpc; 11pt/50 lpc; 12pt/48 lpc
 \IEEEquantizetextheight{c}
 % tweak top margin so that the error is shared equally at the top and bottom
 \IEEEsettopmargin{q}{0sp}
\fi


% compsoc text sizes, margins and spacings
\ifCLASSOPTIONcompsoc
 \columnsep         12bp
 % CS specs for \textwdith are 6.875in
 % \textwidth         6.875in
 % however, measurements from proofs show they are using 3.5in columns
 \textwidth         7in
 \advance\textwidth by \columnsep
 % set the side margins to center the text
 \IEEEsetsidemargin{c}{0pt}
 % top/bottom margins to center
 % could just set \textheight to 9.75in for all the different paper sizes
 % and then quantize, but we'll do it the long way here to allow for easy
 % future per-paper size adjustments
 \IEEEsettextheight{0.625in}{0.625in}% 11in - 2 * 0.625in = 9.75in is the standard text height for compsoc journals
 \IEEEsettopmargin{t}{0.625in}
 \if@IEEEusingcspaper
   \IEEEsettextheight{0.5in}{0.5in}% 10.75in - 2 * 0.5in = 9.75in
   \IEEEsettopmargin{t}{0.5in}
 \fi
 \if@IEEEusingAfourpaper
   \IEEEsettextheight{24.675mm}{24.675mm}% 297mm - 2 * 24.675mm = 247.650mm (9.75in)
   \IEEEsettopmargin{t}{24.675mm}
 \fi
 % tweak textheight to a perfect integer number of lines/page.
 % standard is: 9pt/65 lpc; 10pt/61 lpc; 11pt/53 lpc; 12pt/49 lpc
 \IEEEquantizetextheight{c}
 % tweak top margin so that the error is shared equally at the top and bottom
 \IEEEsettopmargin{q}{0sp}

% compsoc conference
 \ifCLASSOPTIONconference
  % compsoc conference use a larger value for columnsep
  \columnsep 0.25in
  \IEEEsettextwidth{0.75in}{0.75in}
  % set the side margins to center the text (0.75in for letterpaper)
  \IEEEsetsidemargin{c}{0pt}
  % compsoc conferences want 1in top and bottom margin
  \IEEEsettextheight{1in}{1in}
  \IEEEsettopmargin{t}{1in}
  % tweak textheight to a perfect integer number of lines/page.
  % standard is: 9pt/58 lpc; 10pt/53 lpc; 11pt/48 lpc; 12pt/46 lpc
  \IEEEquantizetextheight{c}
  % tweak top margin so that the error is shared equally at the top and bottom
  \IEEEsettopmargin{q}{0sp}
 \fi
\fi



% draft mode settings override that of all other modes
% provides a nice 1in margin all around the paper and extra
% space between the lines for editor's comments
\ifCLASSOPTIONdraftcls
  % we want 1in side margins regardless of paper type
  \IEEEsettextwidth{1in}{1in}
  \IEEEsetsidemargin{c}{0pt}
  % want 1in top and bottom margins
  \IEEEsettextheight{1in}{1in}
  \IEEEsettopmargin{t}{1in}
  % digitize textheight to be an integer number of lines.
  % this may cause the top and bottom margins to be off a tad
  \IEEEquantizetextheight{c}
  % tweak top margin so that the error is shared equally at the top and bottom
  \IEEEsettopmargin{q}{0sp}
\fi



% process CLASSINPUT inner/outer margin
% if inner margin defined, but outer margin not, set outer to inner.
\ifx\CLASSINPUTinnersidemargin\@IEEEundefined
\else
  \ifx\CLASSINPUToutersidemargin\@IEEEundefined
    \edef\CLASSINPUToutersidemargin{\CLASSINPUTinnersidemargin}
  \fi
\fi

\ifx\CLASSINPUToutersidemargin\@IEEEundefined
\else
  % if outer margin defined, but inner margin not, set inner to outer.
  \ifx\CLASSINPUTinnersidemargin\@IEEEundefined
    \edef\CLASSINPUTinnersidemargin{\CLASSINPUToutersidemargin}
  \fi
  \IEEEsettextwidth{\CLASSINPUTinnersidemargin}{\CLASSINPUToutersidemargin}
  \IEEEsetsidemargin{i}{\CLASSINPUTinnersidemargin}
  \typeout{** ATTENTION: Overriding inner side margin to \CLASSINPUTinnersidemargin\space and 
           outer side margin to \CLASSINPUToutersidemargin\space via \string\CLASSINPUT.}
\fi



% process CLASSINPUT top/bottom text margin
% if toptext margin defined, but bottomtext margin not, set bottomtext to toptext margin
\ifx\CLASSINPUTtoptextmargin\@IEEEundefined
\else
  \ifx\CLASSINPUTbottomtextmargin\@IEEEundefined
    \edef\CLASSINPUTbottomtextmargin{\CLASSINPUTtoptextmargin}
  \fi
\fi

\ifx\CLASSINPUTbottomtextmargin\@IEEEundefined
\else
  % if bottomtext margin defined, but toptext margin not, set toptext to bottomtext margin
  \ifx\CLASSINPUTtoptextmargin\@IEEEundefined
    \edef\CLASSINPUTtoptextmargin{\CLASSINPUTbottomtextmargin}
  \fi
  \IEEEsettextheight{\CLASSINPUTtoptextmargin}{\CLASSINPUTbottomtextmargin}
  \IEEEsettopmargin{t}{\CLASSINPUTtoptextmargin}
  \typeout{** ATTENTION: Overriding top text margin to \CLASSINPUTtoptextmargin\space and 
           bottom text margin to \CLASSINPUTbottomtextmargin\space via \string\CLASSINPUT.}
\fi



% default to center header and footer text in the margins
\IEEEsetheadermargin{c}{0pt}
\IEEEsetfootermargin{c}{0pt}

% adjust header and footer positions for compsoc journals
\ifCLASSOPTIONcompsoc
  \ifCLASSOPTIONjournal
    \IEEEsetheadermargin{b}{\@IEEEnormalsizeunitybaselineskip}
    \IEEEsetfootermargin{t}{\@IEEEnormalsizeunitybaselineskip}
  \fi
\fi


% V1.8a display lines per column info message on user's console
\def\IEEEdisplayinfolinespercolumn{\@IEEEtrantmpdimenA=\textheight
% topskip represents only one line even if > baselineskip
\advance\@IEEEtrantmpdimenA by -1\topskip
\@IEEEtrantmpcountA=\@IEEEtrantmpdimenA
\@IEEEtrantmpcountB=\@IEEEtrantmpdimenA
\divide\@IEEEtrantmpcountB by \baselineskip
% need to add one line to include topskip (first) line
\advance\@IEEEtrantmpcountB by 1
% save lines per column value as text
\edef\@IEEEnumlinespercolumninfotxt{\the\@IEEEtrantmpcountB}
% backout topskip advance to allow direct \@IEEEtrantmpcountA comparison
\advance\@IEEEtrantmpcountB by -1
% restore value as text height (without topskip) rather than just as number of lines
\multiply\@IEEEtrantmpcountB by \baselineskip
% is the column height an integer number of lines per column?
\ifnum\@IEEEtrantmpcountA=\@IEEEtrantmpcountB
\edef\@IEEEnumlinespercolumnexactinfotxt{exact}
\else
\@IEEEtrantmpdimenA\@IEEEtrantmpcountA sp\relax
\advance\@IEEEtrantmpdimenA by -\@IEEEtrantmpcountB sp\relax
\edef\@IEEEnumlinespercolumnexactinfotxt{approximate, difference = \the\@IEEEtrantmpdimenA}
\fi
\typeout{-- Lines per column: \@IEEEnumlinespercolumninfotxt\space (\@IEEEnumlinespercolumnexactinfotxt).}}
% delay execution till start of document to allow for user changes
\AtBeginDocument{\IEEEdisplayinfolinespercolumn}



% LIST SPACING CONTROLS

% Controls the amount of EXTRA spacing
% above and below \trivlist 
% Both \list and IED lists override this.
% However, \trivlist will use this as will most
% things built from \trivlist like the \center
% environment.
\topsep           0.5\baselineskip

% Controls the additional spacing around lists preceded
% or followed by blank lines. the IEEE does not increase
% spacing before or after paragraphs so it is set to zero.
% \z@ is the same as zero, but faster.
\partopsep          \z@

% Controls the spacing between paragraphs in lists. 
% The IEEE does not increase spacing before or after paragraphs
% so this is also zero. 
% With IEEEtran.cls, global changes to
% this value DO affect lists (but not IED lists).
\parsep             \z@

% Controls the extra spacing between list items. 
% The IEEE does not put extra spacing between items.
% With IEEEtran.cls, global changes to this value DO affect
% lists (but not IED lists).
\itemsep            \z@

% \itemindent is the amount to indent the FIRST line of a list
% item. It is auto set to zero within the \list environment. To alter
% it, you have to do so when you call the \list.
% However, the IEEE uses this for the theorem environment
% There is an alternative value for this near \leftmargini below
\itemindent         -1em

% \leftmargin, the spacing from the left margin of the main text to
% the left of the main body of a list item is set by \list.
% Hence this statement does nothing for lists.
% But, quote and verse do use it for indention.
\leftmargin         2em

% we retain this stuff from the older IEEEtran.cls so that \list
% will work the same way as before. However, itemize, enumerate and
% description (IED) could care less about what these are as they
% all are overridden.
\leftmargini        2em
%\itemindent         2em  % Alternative values: sometimes used.
%\leftmargini        0em
\leftmarginii       1em
\leftmarginiii    1.5em
\leftmarginiv     1.5em
\leftmarginv      1.0em
\leftmarginvi     1.0em
\labelsep         0.5em 
\labelwidth         \z@


% The old IEEEtran.cls behavior of \list is retained.
% However, the new V1.3 IED list environments override all the
% @list stuff (\@listX is called within \list for the
% appropriate level just before the user's list_decl is called). 
% \topsep is now 2pt as the IEEE puts a little extra space around
% lists - used by those non-IED macros that depend on \list.
% Note that \parsep and \itemsep are not redefined as in 
% the sizexx.clo \@listX (which article.cls uses) so global changes
% of these values DO affect \list
% 
\def\@listi{\leftmargin\leftmargini \topsep 2pt plus 1pt minus 1pt}
\let\@listI\@listi
\def\@listii{\leftmargin\leftmarginii\labelwidth\leftmarginii%
    \advance\labelwidth-\labelsep \topsep 2pt}
\def\@listiii{\leftmargin\leftmarginiii\labelwidth\leftmarginiii%
    \advance\labelwidth-\labelsep \topsep 2pt}
\def\@listiv{\leftmargin\leftmarginiv\labelwidth\leftmarginiv%
    \advance\labelwidth-\labelsep \topsep 2pt}
\def\@listv{\leftmargin\leftmarginv\labelwidth\leftmarginv%
    \advance\labelwidth-\labelsep \topsep 2pt}
\def\@listvi{\leftmargin\leftmarginvi\labelwidth\leftmarginvi%
    \advance\labelwidth-\labelsep \topsep 2pt}


% The IEEE uses 5) not 5.
\def\labelenumi{\theenumi)}     \def\theenumi{\arabic{enumi}}

% The IEEE uses a) not (a)
\def\labelenumii{\theenumii)}  \def\theenumii{\alph{enumii}}

% The IEEE uses iii) not iii.
\def\labelenumiii{\theenumiii)} \def\theenumiii{\roman{enumiii}}

% The IEEE uses A) not A.
\def\labelenumiv{\theenumiv)}   \def\theenumiv{\Alph{enumiv}}

% exactly the same as in article.cls
\def\p@enumii{\theenumi}
\def\p@enumiii{\theenumi(\theenumii)}
\def\p@enumiv{\p@enumiii\theenumiii}

% itemized list label styles
\def\labelitemi{$\scriptstyle\bullet$}
\def\labelitemii{\textbf{--}}
\def\labelitemiii{$\ast$}
\def\labelitemiv{$\cdot$}



% **** V1.3 ENHANCEMENTS ****
% Itemize, Enumerate and Description (IED) List Controls
% ***************************
% 
% 
% The IEEE seems to use at least two different values by
% which ITEMIZED list labels are indented to the right
% For The Journal of Lightwave Technology (JLT) and The Journal
% on Selected Areas in Communications (JSAC), they tend to use
% an indention equal to \parindent. For Transactions on Communications
% they tend to indent ITEMIZED lists a little more--- 1.3\parindent.
% We'll provide both values here for you so that you can choose 
% which one you like in your document using a command such as:
% setlength{\IEEEilabelindent}{\IEEEilabelindentB}
\newdimen\IEEEilabelindentA
\IEEEilabelindentA \parindent

\newdimen\IEEEilabelindentB
\IEEEilabelindentB 1.3\parindent
% However, we'll default to using \parindent
% which makes more sense to me
\newdimen\IEEEilabelindent
\IEEEilabelindent \IEEEilabelindentA


% This controls the default amount the enumerated list labels
% are indented to the right.
% Normally, this is the same as the paragraph indention
\newdimen\IEEEelabelindent
\IEEEelabelindent \parindent

% This controls the default amount the description list labels
% are indented to the right.
% Normally, this is the same as the paragraph indention
\newdimen\IEEEdlabelindent
\IEEEdlabelindent \parindent

% This is the value actually used within the IED lists.
% The IED environments automatically set its value to
% one of the three values above, so global changes do 
% not have any effect
\newdimen\IEEElabelindent
\IEEElabelindent \parindent

% The actual amount labels will be indented is
% \IEEElabelindent multiplied by the factor below
% corresponding to the level of nesting depth
% This provides a means by which the user can
% alter the effective \IEEElabelindent for deeper
% levels
% There may not be such a thing as correct "standard IEEE"
% values. What the IEEE actually does may depend on the specific
% circumstances.
% The first list level almost always has full indention.
% The second levels I've seen have only 75% of the normal indentation
% Three level or greater nestings are very rare. I am guessing
% that they don't use any indentation.
\def\IEEElabelindentfactori{1.0}   % almost always one
\def\IEEElabelindentfactorii{0.75} % 0.0 or 1.0 may be used in some cases
\def\IEEElabelindentfactoriii{0.0} % 0.75? 0.5? 0.0?
\def\IEEElabelindentfactoriv{0.0}
\def\IEEElabelindentfactorv{0.0}
\def\IEEElabelindentfactorvi{0.0}

% value actually used within IED lists, it is auto
% set to one of the 6 values above
% global changes here have no effect
\def\IEEElabelindentfactor{1.0}

% This controls the default spacing between the end of the IED
% list labels and the list text, when normal text is used for
% the labels.
% compsoc uses a larger value here, but we'll set that later
% in the class so that this code block area can be extracted
% as-is for IEEEtrantools.sty
\newdimen\IEEEiednormlabelsep
\IEEEiednormlabelsep 0.6em

% This controls the default spacing between the end of the IED
% list labels and the list text, when math symbols are used for
% the labels (nomenclature lists). The IEEE usually increases the 
% spacing in these cases
\newdimen\IEEEiedmathlabelsep
\IEEEiedmathlabelsep 1.2em

% This controls the extra vertical separation put above and
% below each IED list. the IEEE usually puts a little extra spacing
% around each list. However, this spacing is barely noticeable.
% compsoc uses a larger value here, but we'll set that later
% in the class so that this code block area can be extracted
% as-is for IEEEtrantools.sty
\newskip\IEEEiedtopsep
\IEEEiedtopsep 2pt plus 1pt minus 1pt


% This command is executed within each IED list environment
% at the beginning of the list. You can use this to set the 
% parameters for some/all your IED list(s) without disturbing 
% global parameters that affect things other than lists.
% i.e., renewcommand{\IEEEiedlistdecl}{\setlength{\labelsep}{5em}}
% will alter the \labelsep for the next list(s) until 
% \IEEEiedlistdecl is redefined. 
\def\IEEEiedlistdecl{\relax}

% This command provides an easy way to set \leftmargin based
% on the \labelwidth, \labelsep and the argument \IEEElabelindent
% Usage: \IEEEcalcleftmargin{width-to-indent-the-label}
% output is in the \leftmargin variable, i.e., effectively:
% \leftmargin = argument + \labelwidth + \labelsep
% Note controlled spacing here, shield end of lines with %
\def\IEEEcalcleftmargin#1{\setlength{\leftmargin}{#1}%
\addtolength{\leftmargin}{\labelwidth}%
\addtolength{\leftmargin}{\labelsep}}

% This command provides an easy way to set \labelwidth to the
% width of the given text. It is the same as
% \settowidth{\labelwidth}{label-text}
% and useful as a shorter alternative.
% Typically used to set \labelwidth to be the width
% of the longest label in the list
\def\IEEEsetlabelwidth#1{\settowidth{\labelwidth}{#1}}

% When this command is executed, IED lists will use the 
% IEEEiedmathlabelsep label separation rather than the normal
% spacing. To have an effect, this command must be executed via
% the \IEEEiedlistdecl or within the option of the IED list
% environments.
\def\IEEEusemathlabelsep{\setlength{\labelsep}{\IEEEiedmathlabelsep}}

% A flag which controls whether the IED lists automatically
% calculate \leftmargin from \IEEElabelindent, \labelwidth and \labelsep
% Useful if you want to specify your own \leftmargin
% This flag must be set (\IEEEnocalcleftmargintrue or \IEEEnocalcleftmarginfalse) 
% via the \IEEEiedlistdecl or within the option of the IED list
% environments to have an effect.
\newif\ifIEEEnocalcleftmargin
\IEEEnocalcleftmarginfalse

% A flag which controls whether \IEEElabelindent is multiplied by
% the \IEEElabelindentfactor for each list level.
% This flag must be set via the \IEEEiedlistdecl or within the option 
% of the IED list environments to have an effect.
\newif\ifIEEEnolabelindentfactor
\IEEEnolabelindentfactorfalse


% internal variable to indicate type of IED label
% justification
% 0 - left; 1 - center; 2 - right
\def\@IEEEiedjustify{0}


% commands to allow the user to control IED
% label justifications. Use these commands within
% the IED environment option or in the \IEEEiedlistdecl
% Note that changing the normal list justifications
% is nonstandard and the IEEE may not like it if you do so!
% I include these commands as they may be helpful to
% those who are using these enhanced list controls for
% other non-IEEE related LaTeX work.
% itemize and enumerate automatically default to right
% justification, description defaults to left.
\def\IEEEiedlabeljustifyl{\def\@IEEEiedjustify{0}}%left
\def\IEEEiedlabeljustifyc{\def\@IEEEiedjustify{1}}%center
\def\IEEEiedlabeljustifyr{\def\@IEEEiedjustify{2}}%right




% commands to save to and restore from the list parameter copies
% this allows us to set all the list parameters within
% the list_decl and prevent \list (and its \@list) 
% from overriding any of our parameters
% V1.6 use \edefs instead of dimen's to conserve dimen registers
% Note controlled spacing here, shield end of lines with %
\def\@IEEEsavelistparams{\edef\@IEEEiedtopsep{\the\topsep}%
\edef\@IEEEiedlabelwidth{\the\labelwidth}%
\edef\@IEEEiedlabelsep{\the\labelsep}%
\edef\@IEEEiedleftmargin{\the\leftmargin}%
\edef\@IEEEiedpartopsep{\the\partopsep}%
\edef\@IEEEiedparsep{\the\parsep}%
\edef\@IEEEieditemsep{\the\itemsep}%
\edef\@IEEEiedrightmargin{\the\rightmargin}%
\edef\@IEEEiedlistparindent{\the\listparindent}%
\edef\@IEEEieditemindent{\the\itemindent}}

% Note controlled spacing here
\def\@IEEErestorelistparams{\topsep\@IEEEiedtopsep\relax%
\labelwidth\@IEEEiedlabelwidth\relax%
\labelsep\@IEEEiedlabelsep\relax%
\leftmargin\@IEEEiedleftmargin\relax%
\partopsep\@IEEEiedpartopsep\relax%
\parsep\@IEEEiedparsep\relax%
\itemsep\@IEEEieditemsep\relax%
\rightmargin\@IEEEiedrightmargin\relax%
\listparindent\@IEEEiedlistparindent\relax%
\itemindent\@IEEEieditemindent\relax}


% v1.6b provide original LaTeX IED list environments
% note that latex.ltx defines \itemize and \enumerate, but not \description
% which must be created by the base classes
% save original LaTeX itemize and enumerate
\let\LaTeXitemize\itemize
\let\endLaTeXitemize\enditemize
\let\LaTeXenumerate\enumerate
\let\endLaTeXenumerate\endenumerate

% provide original LaTeX description environment from article.cls
\newenvironment{LaTeXdescription}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel}}
               {\endlist}
\newcommand*\descriptionlabel[1]{\hspace\labelsep
                                 \normalfont\bfseries #1}


% override LaTeX's default IED lists
\def\itemize{\@IEEEitemize}
\def\enditemize{\@endIEEEitemize}
\def\enumerate{\@IEEEenumerate}
\def\endenumerate{\@endIEEEenumerate}
\def\description{\@IEEEdescription}
\def\enddescription{\@endIEEEdescription}

% provide the user with aliases - may help those using packages that
% override itemize, enumerate, or description
\def\IEEEitemize{\@IEEEitemize}
\def\endIEEEitemize{\@endIEEEitemize}
\def\IEEEenumerate{\@IEEEenumerate}
\def\endIEEEenumerate{\@endIEEEenumerate}
\def\IEEEdescription{\@IEEEdescription}
\def\endIEEEdescription{\@endIEEEdescription}


% V1.6 we want to keep the IEEEtran IED list definitions as our own internal
% commands so they are protected against redefinition
\def\@IEEEitemize{\@ifnextchar[{\@@IEEEitemize}{\@@IEEEitemize[\relax]}}
\def\@IEEEenumerate{\@ifnextchar[{\@@IEEEenumerate}{\@@IEEEenumerate[\relax]}}
\def\@IEEEdescription{\@ifnextchar[{\@@IEEEdescription}{\@@IEEEdescription[\relax]}}
\def\@endIEEEitemize{\endlist}
\def\@endIEEEenumerate{\endlist}
\def\@endIEEEdescription{\endlist}


% DO NOT ALLOW BLANK LINES TO BE IN THESE IED ENVIRONMENTS
% AS THIS WILL FORCE NEW PARAGRAPHS AFTER THE IED LISTS
% IEEEtran itemized list MDS 1/2001
% Note controlled spacing here, shield end of lines with %
\def\@@IEEEitemize[#1]{%
                \ifnum\@itemdepth>3\relax\@toodeep\else%
                \ifnum\@listdepth>5\relax\@toodeep\else%
                \advance\@itemdepth\@ne%
                \edef\@itemitem{labelitem\romannumeral\the\@itemdepth}%
                % get the IEEElabelindentfactor for this level
                \advance\@listdepth\@ne% we need to know what the level WILL be
                \edef\IEEElabelindentfactor{\csname IEEElabelindentfactor\romannumeral\the\@listdepth\endcsname}%
                \advance\@listdepth-\@ne% undo our increment
                \def\@IEEEiedjustify{2}% right justified labels are default
                % set other defaults
                \IEEEnocalcleftmarginfalse%
                \IEEEnolabelindentfactorfalse%
                \topsep\IEEEiedtopsep%
                \IEEElabelindent\IEEEilabelindent%
                \labelsep\IEEEiednormlabelsep%
                \partopsep 0ex%
                \parsep 0ex%
                \itemsep 0ex%
                \rightmargin 0em%
                \listparindent 0em%
                \itemindent 0em%
                % calculate the label width
                % the user can override this later if
                % they specified a \labelwidth
                \settowidth{\labelwidth}{\csname labelitem\romannumeral\the\@itemdepth\endcsname}%
                \@IEEEsavelistparams% save our list parameters
                \list{\csname\@itemitem\endcsname}{%
                \@IEEErestorelistparams% override any list{} changes
                                       % to our globals
                \let\makelabel\@IEEEiedmakelabel% v1.6b setup \makelabel
                \IEEEiedlistdecl% let user alter parameters
                #1\relax%
                % If the user has requested not to use the
                % IEEElabelindent factor, don't revise \IEEElabelindent
                \ifIEEEnolabelindentfactor\relax%
                \else\IEEElabelindent=\IEEElabelindentfactor\IEEElabelindent%
                \fi%
                % Unless the user has requested otherwise,
                % calculate our left margin based
                % on \IEEElabelindent, \labelwidth and
                % \labelsep
                \ifIEEEnocalcleftmargin\relax%
                \else\IEEEcalcleftmargin{\IEEElabelindent}%
                \fi}\fi\fi}%


% DO NOT ALLOW BLANK LINES TO BE IN THESE IED ENVIRONMENTS
% AS THIS WILL FORCE NEW PARAGRAPHS AFTER THE IED LISTS
% IEEEtran enumerate list MDS 1/2001
% Note controlled spacing here, shield end of lines with %
\def\@@IEEEenumerate[#1]{%
                \ifnum\@enumdepth>3\relax\@toodeep\else%
                \ifnum\@listdepth>5\relax\@toodeep\else%
                \advance\@enumdepth\@ne%
                \edef\@enumctr{enum\romannumeral\the\@enumdepth}%
                % get the IEEElabelindentfactor for this level
                \advance\@listdepth\@ne% we need to know what the level WILL be
                \edef\IEEElabelindentfactor{\csname IEEElabelindentfactor\romannumeral\the\@listdepth\endcsname}%
                \advance\@listdepth-\@ne% undo our increment
                \def\@IEEEiedjustify{2}% right justified labels are default
                % set other defaults
                \IEEEnocalcleftmarginfalse%
                \IEEEnolabelindentfactorfalse%
                \topsep\IEEEiedtopsep%
                \IEEElabelindent\IEEEelabelindent%
                \labelsep\IEEEiednormlabelsep%
                \partopsep 0ex%
                \parsep 0ex%
                \itemsep 0ex%
                \rightmargin 0em%
                \listparindent 0em%
                \itemindent 0em%
                % calculate the label width
                % We'll set it to the width suitable for all labels using
                % normalfont 1) to 9)
                % The user can override this later
                \settowidth{\labelwidth}{9)}%
                \@IEEEsavelistparams% save our list parameters
                \list{\csname label\@enumctr\endcsname}{\usecounter{\@enumctr}%
                \@IEEErestorelistparams% override any list{} changes
                                       % to our globals
                \let\makelabel\@IEEEiedmakelabel% v1.6b setup \makelabel
                \IEEEiedlistdecl% let user alter parameters 
                #1\relax%
                % If the user has requested not to use the
                % IEEElabelindent factor, don't revise \IEEElabelindent
                \ifIEEEnolabelindentfactor\relax%
                \else\IEEElabelindent=\IEEElabelindentfactor\IEEElabelindent%
                \fi%
                % Unless the user has requested otherwise,
                % calculate our left margin based
                % on \IEEElabelindent, \labelwidth and
                % \labelsep
                \ifIEEEnocalcleftmargin\relax%
                \else\IEEEcalcleftmargin{\IEEElabelindent}%
                \fi}\fi\fi}%


% DO NOT ALLOW BLANK LINES TO BE IN THESE IED ENVIRONMENTS
% AS THIS WILL FORCE NEW PARAGRAPHS AFTER THE IED LISTS
% IEEEtran description list MDS 1/2001
% Note controlled spacing here, shield end of lines with %
\def\@@IEEEdescription[#1]{%
                \ifnum\@listdepth>5\relax\@toodeep\else%
                % get the IEEElabelindentfactor for this level
                \advance\@listdepth\@ne% we need to know what the level WILL be
                \edef\IEEElabelindentfactor{\csname IEEElabelindentfactor\romannumeral\the\@listdepth\endcsname}%
                \advance\@listdepth-\@ne% undo our increment
                \def\@IEEEiedjustify{0}% left justified labels are default
                % set other defaults
                \IEEEnocalcleftmarginfalse%
                \IEEEnolabelindentfactorfalse%
                \topsep\IEEEiedtopsep% 
                \IEEElabelindent\IEEEdlabelindent%
                % assume normal labelsep
                \labelsep\IEEEiednormlabelsep%
                \partopsep 0ex%
                \parsep 0ex%
                \itemsep 0ex%
                \rightmargin 0em%
                \listparindent 0em%
                \itemindent 0em%
                % Bogus label width in case the user forgets
                % to set it.
                % TIP: If you want to see what a variable's width is you
                % can use the TeX command \showthe\width-variable to 
                % display it on the screen during compilation 
                % (This might be helpful to know when you need to find out
                % which label is the widest)
                \settowidth{\labelwidth}{Hello}%
                \@IEEEsavelistparams% save our list parameters
                \list{}{\@IEEErestorelistparams% override any list{} changes
                                               % to our globals
                \let\makelabel\@IEEEiedmakelabel% v1.6b setup \makelabel
                \IEEEiedlistdecl% let user alter parameters 
                #1\relax%
                % If the user has requested not to use the
                % labelindent factor, don't revise \IEEElabelindent
                \ifIEEEnolabelindentfactor\relax%
                \else\IEEElabelindent=\IEEElabelindentfactor\IEEElabelindent%
                \fi%
                % Unless the user has requested otherwise,
                % calculate our left margin based
                % on \IEEElabelindent, \labelwidth and
                % \labelsep
                \ifIEEEnocalcleftmargin\relax%
                \else\IEEEcalcleftmargin{\IEEElabelindent}\relax%
                \fi}\fi}

% v1.6b we use one makelabel that does justification as needed.
\def\@IEEEiedmakelabel#1{\relax\if\@IEEEiedjustify 0\relax
\makebox[\labelwidth][l]{\normalfont #1}\else
\if\@IEEEiedjustify 1\relax
\makebox[\labelwidth][c]{\normalfont #1}\else
\makebox[\labelwidth][r]{\normalfont #1}\fi\fi}


% compsoc uses a larger value for the normal labelsep
% and also extra spacing above and below each list
\ifCLASSOPTIONcompsoc
  \IEEEiednormlabelsep 1.2em
  \IEEEiedtopsep 6pt plus 3pt minus 3pt
\fi


% VERSE and QUOTE
% V1.7 define environments with newenvironment
\newenvironment{verse}{\let\\=\@centercr
    \list{}{\itemsep\z@ \itemindent -1.5em \listparindent \itemindent
    \rightmargin\leftmargin\advance\leftmargin 1.5em}\item\relax}
    {\endlist}
\newenvironment{quotation}{\list{}{\listparindent 1.5em \itemindent\listparindent
    \rightmargin\leftmargin \parsep 0pt plus 1pt}\item\relax}
    {\endlist}
\newenvironment{quote}{\list{}{\rightmargin\leftmargin}\item\relax}
    {\endlist}


% \titlepage
% provided only for backward compatibility. \maketitle is the correct
% way to create the title page. 
\def\titlepage{\@restonecolfalse\if@twocolumn\@restonecoltrue\onecolumn
    \else \newpage \fi \thispagestyle{empty}\c@page\z@}
\def\endtitlepage{\if@restonecol\twocolumn \else \newpage \fi}

% standard values from article.cls
\arraycolsep     5pt
\arrayrulewidth .4pt
\doublerulesep   2pt

\tabcolsep       6pt
\tabbingsep      0.5em


%% FOOTNOTES
%
%\skip\footins 10pt plus 4pt minus 2pt
% V1.6 respond to changes in font size
% space added above the footnotes (if present)
\skip\footins 0.9\baselineskip  plus 0.4\baselineskip  minus 0.2\baselineskip

% V1.6, we need to make \footnotesep responsive to changes
% in \baselineskip or strange spacings will result when in
% draft mode. Here is a little LaTeX secret - \footnotesep
% determines the height of an invisible strut that is placed
% *above* the baseline of footnotes after the first. Since
% LaTeX considers the space for characters to be 0.7\baselineskip
% above the baseline and 0.3\baselineskip below it, we need to
% use 0.7\baselineskip as a \footnotesep to maintain equal spacing
% between all the lines of the footnotes. The IEEE often uses a tad
% more, so use 0.8\baselineskip. This slightly larger value also helps
% the text to clear the footnote marks. Note that \thanks in IEEEtran
% uses its own value of \footnotesep which is set in \maketitle.
{\footnotesize
\global\footnotesep 0.8\baselineskip}


\skip\@mpfootins = \skip\footins
\fboxsep = 3pt
\fboxrule = .4pt
% V1.6 use 1em, then use LaTeX2e's \@makefnmark
% Note that the IEEE normally *left* aligns the footnote marks, so we don't need
% box resizing tricks here.
\long\def\@makefntext#1{\parindent 1em\indent\hbox{\@makefnmark}#1}% V1.6 use 1em
% V1.7 compsoc does not use superscipts for footnote marks
\ifCLASSOPTIONcompsoc
\def\@IEEEcompsocmakefnmark{\hbox{\normalfont\@thefnmark.\ }}
\long\def\@makefntext#1{\parindent 1em\indent\hbox{\@IEEEcompsocmakefnmark}#1}
\fi

% The IEEE does not use footnote rules
\def\footnoterule{}

% V1.7 for compsoc, the IEEE uses a footnote rule only for \thanks. We devise a "one-shot"
% system to implement this.
\newif\if@IEEEenableoneshotfootnoterule
\@IEEEenableoneshotfootnoterulefalse
\ifCLASSOPTIONcompsoc
\def\footnoterule{\relax\if@IEEEenableoneshotfootnoterule
\kern-5pt
\hbox to \columnwidth{\hfill\vrule width 0.5\columnwidth height 0.4pt\hfill}
\kern4.6pt
\global\@IEEEenableoneshotfootnoterulefalse
\else
\relax
\fi}
\fi

% V1.6 do not allow LaTeX to break a footnote across multiple pages
\interfootnotelinepenalty=10000

% V1.6 discourage breaks within equations
% Note that amsmath normally sets this to 10000,
% but LaTeX2e normally uses 100.
\interdisplaylinepenalty=2500

% default allows section depth up to /paragraph
\setcounter{secnumdepth}{4}

% technotes do not allow /paragraph
\ifCLASSOPTIONtechnote
   \setcounter{secnumdepth}{3}
\fi
% neither do compsoc conferences
\@IEEEcompsocconfonly{\setcounter{secnumdepth}{3}}


\newcounter{section}
\newcounter{subsection}[section]
\newcounter{subsubsection}[subsection]
\newcounter{paragraph}[subsubsection]

% used only by IEEEtran's IEEEeqnarray as other packages may
% have their own, different, implementations
\newcounter{IEEEsubequation}[equation]

% as shown when called by user from \ref, \label and in table of contents
\def\theequation{\arabic{equation}}                          % 1
\def\theIEEEsubequation{\theequation\alph{IEEEsubequation}}  % 1a (used only by IEEEtran's IEEEeqnarray)
\ifCLASSOPTIONcompsoc
% compsoc is all arabic
\def\thesection{\arabic{section}}                
\def\thesubsection{\thesection.\arabic{subsection}}
\def\thesubsubsection{\thesubsection.\arabic{subsubsection}}
\def\theparagraph{\thesubsubsection.\arabic{paragraph}}
\else
\def\thesection{\Roman{section}}                             % I
% V1.7, \mbox prevents breaks around - 
\def\thesubsection{\mbox{\thesection-\Alph{subsection}}}     % I-A
% V1.7 use I-A1 format used by the IEEE rather than I-A.1
\def\thesubsubsection{\thesubsection\arabic{subsubsection}}  % I-A1
\def\theparagraph{\thesubsubsection\alph{paragraph}}         % I-A1a
\fi

% From Heiko Oberdiek. Because of the \mbox in \thesubsection, we need to
% tell hyperref to disable the \mbox command when making PDF bookmarks.
% This done already with hyperref.sty version 6.74o and later, but
% it will not hurt to do it here again for users of older versions.
\@ifundefined{pdfstringdefPreHook}{\let\pdfstringdefPreHook\@empty}{}%
\g@addto@macro\pdfstringdefPreHook{\let\mbox\relax}


% Main text forms (how shown in main text headings)
% V1.6, using \thesection in \thesectiondis allows changes
% in the former to automatically appear in the latter
\ifCLASSOPTIONcompsoc
  \ifCLASSOPTIONconference% compsoc conference
    \def\thesectiondis{\thesection.}
    \def\thesubsectiondis{\thesectiondis\arabic{subsection}.}
    \def\thesubsubsectiondis{\thesubsectiondis\arabic{subsubsection}.}
    \def\theparagraphdis{\thesubsubsectiondis\arabic{paragraph}.}
  \else% compsoc not conferencs
    \def\thesectiondis{\thesection}
    \def\thesubsectiondis{\thesectiondis.\arabic{subsection}}
    \def\thesubsubsectiondis{\thesubsectiondis.\arabic{subsubsection}}
    \def\theparagraphdis{\thesubsubsectiondis.\arabic{paragraph}}
  \fi
\else% not compsoc
  \def\thesectiondis{\thesection.}                   % I.
  \def\thesubsectiondis{\Alph{subsection}.}          % B.
  \def\thesubsubsectiondis{\arabic{subsubsection})}  % 3)
  \def\theparagraphdis{\alph{paragraph})}            % d)
\fi

% just like LaTeX2e's \@eqnnum
\def\theequationdis{{\normalfont \normalcolor (\theequation)}}% (1)
% IEEEsubequation used only by IEEEtran's IEEEeqnarray
\def\theIEEEsubequationdis{{\normalfont \normalcolor (\theIEEEsubequation)}}% (1a)
% redirect LaTeX2e's equation number display and all that depend on
% it, through IEEEtran's \theequationdis
\def\@eqnnum{\theequationdis}



% V1.7 provide string macros as article.cls does
\def\contentsname{Contents}
\def\listfigurename{List of Figures}
\def\listtablename{List of Tables}
\def\refname{References}
\def\indexname{Index}
\def\figurename{Fig.}
\def\tablename{TABLE}
\@IEEEcompsocconfonly{\def\figurename{Figure}}
\def\partname{Part}
\def\appendixname{Appendix}
\def\abstractname{Abstract}
% IEEE specific names
\def\IEEEkeywordsname{Index Terms}
\def\IEEEproofname{Proof}


% LIST OF FIGURES AND TABLES AND TABLE OF CONTENTS
%
\def\@pnumwidth{1.55em}
\def\@tocrmarg{2.55em}
\def\@dotsep{4.5}
\setcounter{tocdepth}{3}

% adjusted some spacings here so that section numbers will not easily 
% collide with the section titles. 
% VIII; VIII-A; and VIII-A.1 are usually the worst offenders.
% MDS 1/2001
\def\tableofcontents{\section*{\contentsname}\@starttoc{toc}}
\def\l@section#1#2{\addpenalty{\@secpenalty}\addvspace{1.0em plus 1pt}%
    \@tempdima 2.75em \begingroup \parindent \z@ \rightskip \@pnumwidth%
    \parfillskip-\@pnumwidth {\bfseries\leavevmode #1}\hfil\hbox to\@pnumwidth{\hss #2}\par%
    \endgroup}
% argument format #1:level, #2:labelindent,#3:labelsep
\def\l@subsection{\@dottedtocline{2}{2.75em}{3.75em}}
\def\l@subsubsection{\@dottedtocline{3}{6.5em}{4.5em}}
% must provide \l@ defs for ALL sublevels EVEN if tocdepth
% is such as they will not appear in the table of contents
% these defs are how TOC knows what level these things are!
\def\l@paragraph{\@dottedtocline{4}{6.5em}{5.5em}}
\def\l@subparagraph{\@dottedtocline{5}{6.5em}{6.5em}}
\def\listoffigures{\section*{\listfigurename}\@starttoc{lof}}
\def\l@figure{\@dottedtocline{1}{0em}{2.75em}}
\def\listoftables{\section*{\listtablename}\@starttoc{lot}}
\let\l@table\l@figure


% Definitions for floats
%
% Normal Floats
% V1.8 floatsep et al. revised down by 0.15\baselineskip
% to account for the sideeffects of \topskip compensation
\floatsep 0.85\baselineskip plus  0.2\baselineskip minus  0.2\baselineskip
\textfloatsep 1.55\baselineskip plus  0.2\baselineskip minus  0.4\baselineskip
\@fptop 0pt plus 1fil
\@fpsep 0.75\baselineskip plus 2fil
\@fpbot 0pt plus 1fil
\def\topfraction{0.9}
\def\bottomfraction{0.4}
\def\floatpagefraction{0.8}
% V1.7, let top floats approach 90% of page
\def\textfraction{0.1}

% Double Column Floats
\dblfloatsep 0.85\baselineskip plus  0.2\baselineskip minus  0.2\baselineskip

\dbltextfloatsep 1.55\baselineskip plus  0.2\baselineskip minus  0.4\baselineskip
% Note that it would be nice if the rubber here actually worked in LaTeX2e.
% There is a long standing limitation in LaTeX, first discovered (to the best
% of my knowledge) by Alan Jeffrey in 1992. LaTeX ignores the stretchable
% portion of \dbltextfloatsep, and as a result, double column figures can and
% do result in an non-integer number of lines in the main text columns with
% underfull vbox errors as a consequence. A post to comp.text.tex
% by Donald Arseneau confirms that this had not yet been fixed in 1998.
% IEEEtran V1.6 will fix this problem for you in the titles, but it doesn't
% protect you from other double floats. Happy vspace'ing.

\@dblfptop 0pt plus 1fil
\@dblfpsep 0.75\baselineskip plus 2fil
\@dblfpbot 0pt plus 1fil
\def\dbltopfraction{0.8}
\def\dblfloatpagefraction{0.8}
\setcounter{dbltopnumber}{4}

\intextsep 0.85\baselineskip plus 0.2\baselineskip minus  0.2\baselineskip
\setcounter{topnumber}{2}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}



% article class provides these, we should too.
\newlength\abovecaptionskip
\newlength\belowcaptionskip
% but only \abovecaptionskip is used above figure captions and *below* table
% captions
\setlength\abovecaptionskip{0.5\baselineskip}
% compsoc journals are a little more generous
\ifCLASSOPTIONcompsoc\ifCLASSOPTIONjournal
  \setlength\abovecaptionskip{0.75\baselineskip}
\fi\fi
\setlength\belowcaptionskip{0pt}
% V1.6 create hooks in case the caption spacing ever needs to be
% overridden by a user
\def\@IEEEfigurecaptionsepspace{\vskip\abovecaptionskip\relax}%
\def\@IEEEtablecaptionsepspace{\vskip\abovecaptionskip\relax}%


% 1.6b revise caption system so that \@makecaption uses two arguments
% as with LaTeX2e. Otherwise, there will be problems when using hyperref.
\def\@IEEEtablestring{table}


% V1.8 compensate for \topskip so top of top figures align with tops of the first lines of main text
% here we calculate a space equal to the amount \topskip exceeds the main text height
% we hook in at \@floatboxreset
\def\@IEEEfiguretopskipspace{\ifdim\prevdepth=-1000pt\relax
\setlength{\@IEEEtrantmpdimenA}{1\topskip}\relax
\addtolength{\@IEEEtrantmpdimenA}{-0.7\@IEEEnormalsizeunitybaselineskip}\relax
\vspace*{\@IEEEtrantmpdimenA}\fi}
% V1.8 compensate for \topskip at the top of top tables so caption text is on main text baseline
% use a strut set on the caption baseline within \@makecaption
\def\@IEEEtabletopskipstrut{\ifdim\prevdepth=-1000pt\rule{0pt}{\topskip}\fi}
% the \ifdim\prevdepth checks are always expected to be true for IEEE style float caption ordering
% because top of figure content and top of captions in tables is the first thing on the vertical
% list of these floats
% thanks to Donald Arseneau for his 2000/11/11 post "Re: caption hacking" with info on this topic.


\ifCLASSOPTIONcompsoc
% V1.7 compsoc \@makecaption
\ifCLASSOPTIONconference% compsoc conference
\long\def\@makecaption#1#2{%
% test if is a for a figure or table
\ifx\@captype\@IEEEtablestring%
% if a table, do table caption
\footnotesize\bgroup\par\centering\@IEEEtabletopskipstrut{\normalfont\footnotesize {#1.}\nobreakspace\scshape #2}\par\addvspace{0.5\baselineskip}\egroup%
\@IEEEtablecaptionsepspace
% if not a table, format it as a figure
\else
\@IEEEfigurecaptionsepspace
\setbox\@tempboxa\hbox{\normalfont\footnotesize {#1.}\nobreakspace #2}%
\ifdim \wd\@tempboxa >\hsize%
% if caption is longer than a line, let it wrap around
\setbox\@tempboxa\hbox{\normalfont\footnotesize {#1.}\nobreakspace}%
\parbox[t]{\hsize}{\normalfont\footnotesize \noindent\unhbox\@tempboxa#2}%
% if caption is shorter than a line, center
\else%
\hbox to\hsize{\normalfont\footnotesize\hfil\box\@tempboxa\hfil}%
\fi\fi}
%
\else% nonconference compsoc
\long\def\@makecaption#1#2{%
% test if is a for a figure or table
\ifx\@captype\@IEEEtablestring%
% if a table, do table caption
\footnotesize\bgroup\par\centering\@IEEEtabletopskipstrut{\normalfont\sffamily\footnotesize #1}\\{\normalfont\sffamily\footnotesize #2}\par\addvspace{0.5\baselineskip}\egroup%
\@IEEEtablecaptionsepspace
% if not a table, format it as a figure
\else
\@IEEEfigurecaptionsepspace
\setbox\@tempboxa\hbox{\normalfont\sffamily\footnotesize {#1.}\nobreakspace #2}%
\ifdim \wd\@tempboxa >\hsize%
% if caption is longer than a line, let it wrap around
\setbox\@tempboxa\hbox{\normalfont\sffamily\footnotesize {#1.}\nobreakspace}%
\parbox[t]{\hsize}{\normalfont\sffamily\footnotesize \noindent\unhbox\@tempboxa#2}%
% if caption is shorter than a line, left justify
\else%
\hbox to\hsize{\normalfont\sffamily\footnotesize\box\@tempboxa\hfil}%
\fi\fi}
\fi
%
\else% traditional noncompsoc \@makecaption
\long\def\@makecaption#1#2{%
% test if is a for a figure or table
\ifx\@captype\@IEEEtablestring%
% if a table, do table caption
\footnotesize\bgroup\par\centering\@IEEEtabletopskipstrut{\normalfont\footnotesize #1}\\{\normalfont\footnotesize\scshape #2}\par\addvspace{0.5\baselineskip}\egroup%
\@IEEEtablecaptionsepspace
% if not a table, format it as a figure
\else
\@IEEEfigurecaptionsepspace
% 3/2001 use footnotesize, not small; use two nonbreaking spaces, not one
\setbox\@tempboxa\hbox{\normalfont\footnotesize {#1.}\nobreakspace\nobreakspace #2}%
\ifdim \wd\@tempboxa >\hsize%
% if caption is longer than a line, let it wrap around
\setbox\@tempboxa\hbox{\normalfont\footnotesize {#1.}\nobreakspace\nobreakspace}%
\parbox[t]{\hsize}{\normalfont\footnotesize\noindent\unhbox\@tempboxa#2}%
% if caption is shorter than a line, center if conference, left justify otherwise
\else%
\ifCLASSOPTIONconference \hbox to\hsize{\normalfont\footnotesize\hfil\box\@tempboxa\hfil}%
\else \hbox to\hsize{\normalfont\footnotesize\box\@tempboxa\hfil}%
\fi\fi\fi}
\fi



% V1.7 disable captions class option, do so in a way that retains operation of \label
% within \caption
\ifCLASSOPTIONcaptionsoff
\long\def\@makecaption#1#2{\vspace*{2em}\footnotesize\bgroup\par\addvspace{0.5\baselineskip}\centering{\footnotesize #1}\par\addvspace{0.5\baselineskip}\egroup%
\let\@IEEEtemporiglabeldefsave\label
\let\@IEEEtemplabelargsave\relax
\def\label##1{\gdef\@IEEEtemplabelargsave{##1}}%
\setbox\@tempboxa\hbox{#2}%
\let\label\@IEEEtemporiglabeldefsave
\ifx\@IEEEtemplabelargsave\relax\else\label{\@IEEEtemplabelargsave}\fi}
\fi


% V1.7 define end environments with \def not \let so as to work OK with
% preview-latex
\newcounter{figure}
\def\thefigure{\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename\nobreakspace\thefigure}
% V1.8 within figures add \@IEEEfiguretopskipspace compensation to LaTeX2e's \@floatboxreset
\def\figure{\def\@floatboxreset{\reset@font\normalsize\@setminipage\@IEEEfiguretopskipspace}\@float{figure}}
\def\endfigure{\end@float}
% V1.8 also add \@IEEEfiguretopskipspace compensation to \figure*
\@namedef{figure*}{\def\@floatboxreset{\reset@font\normalsize\@setminipage\@IEEEfiguretopskipspace}\@dblfloat{figure}}
\@namedef{endfigure*}{\end@dblfloat}

\newcounter{table}
\ifCLASSOPTIONcompsoc
\def\thetable{\arabic{table}}
\else
\def\thetable{\@Roman\c@table}
\fi
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename\nobreakspace\thetable}
% V1.6 The IEEE uses 8pt text for tables
% within tables alter LaTeX2e's \@floatboxreset to use \footnotesize
\def\table{\def\@floatboxreset{\reset@font\footnotesize\@setminipage}\@float{table}}
\def\endtable{\end@float}
% v1.6b double column tables need to default to footnotesize as well.
\@namedef{table*}{\def\@floatboxreset{\reset@font\footnotesize\@setminipage}\@dblfloat{table}}
\@namedef{endtable*}{\end@dblfloat}




%% -- Command Argument Scanning Support Functions --
%% V1.8a 

% usage: \@IEEEstripouterbraces*{}
% \@IEEEstripouterbraces fully expands its argument (which it then stores
% in \@IEEEstripouterbracesarg) via \edef, then removes any outer enclosing
% braces, and finally stores the result in the macro
% \@IEEEstrippedouterbraces.
%
% For example:
% \@IEEEstripouterbraces{{{{ab}c}}}
% results in:
%
% \@IEEEstripouterbracesarg ==> a macro containing {{{ab}c}}
% \@IEEEstrippedouterbraces ==> a macro containing {ab}c
%
% the *-star form,\@IEEEstripouterbraces*, does not expand the argument
% contents during processing
\def\@IEEEstripouterbraces{\@ifstar{\let\@IEEEstripouterbracesdef=\def\@@IEEEstripouterbraces}{\let\@IEEEstripouterbracesdef=\edef\@@IEEEstripouterbraces}}

\def\@@IEEEstripouterbraces#1{\@IEEEstripouterbracesdef\@IEEEstripouterbracesarg{#1}\relax
% If the macro is unchanged after being acquired as a single delimited
% argument, we know we have one sequence of tokens without any enclosing
% braces. Loop until this is true.
\loop
  \expandafter\@@@IEEEstripouterbraces\@IEEEstripouterbracesarg\@IEEEgeneralsequenceDELIMITER
\ifx\@IEEEstrippedouterbraces\@IEEEstripouterbracesarg
\else
  \let\@IEEEstripouterbracesarg\@IEEEstrippedouterbraces
\repeat}

\def\@@@IEEEstripouterbraces#1\@IEEEgeneralsequenceDELIMITER{\def\@IEEEstrippedouterbraces{#1}}



% usage: \@IEEEextractgroup*{}
% \@IEEEextractgroup fully expands its argument (which it then stores in
% \@IEEEextractgrouparg) via \edef and then assigns the first "brace group"
% of tokens to the macro \@IEEEextractedgroup.
% The remaining groups, if any, are stored in the macro
% \@IEEEextractedgroupremain. If the argument does not contain the requisite
% groups, the respective macros will be defined to be empty.
% There is an asymmetry in that \@IEEEextractedgroup is stripped of its first
% outer grouping while \@IEEEextractedgroupremain retains even the outer
% grouping (if present) that originally identified it as a group.
%
% For example:
% \@IEEEextractgroup{{{ab}}{c{de}}}
% results in:
%
% \@IEEEextractgrouparg       ==> a macro containing {{ab}}{c{de}}
% \@IEEEextractedgroup        ==> a macro containing {ab}
% \@IEEEextractedgroupremain  ==> a macro containing {c{de}}
%
% The *-star form, \@IEEEextractgroup*, does not expand its argument
% contents during processing.
\def\@IEEEextractgroup{\@ifstar{\let\@IEEEextractgroupdef=\def\@@IEEEextractgroup}{\let\@IEEEextractgroupdef=\edef\@@IEEEextractgroup}}

\def\@@IEEEextractgroup#1{\@IEEEextractgroupdef\@IEEEextractgrouparg{#1}\relax
% trap the case of an empty extracted group as this would cause problems with
% \@IEEEextractgroupremain's argument acquisition
\ifx\@IEEEextractgrouparg\@empty
  \def\@IEEEextractedgroup{}\relax
  \def\@IEEEextractedgroupremain{}\relax
\else
  % We have to use some dirty tricks here. We want to insert {} around
  % whatever remains after the first group so that TeX's argument scanner
  % will preserve any originally enclosing braces as well as provide an
  % empty argument to acquire even if there isn't a second group.
  % In this first of two dirty tricks, we put a } at the end of the structure
  % we are going to extract from. The \ifnum0=`{\fi keeps TeX happy to allow
  % what would otherwise be an unbalanced macro definition for
  % \@@IEEEextractgroup to be acceptable to it.
  \ifnum0=`{\fi\expandafter\@IEEEextractgroupremain\@IEEEextractgrouparg}\relax
\fi}

% In the second part of the dirty tricks, we insert a leading { right after
% the first group is acquired, but before the remainder is. Again, the
% \ifnum0=`}\fi keeps TeX happy during definition time, but will disappear
% during run time.
\def\@IEEEextractgroupremain#1{\def\@IEEEextractedgroup{#1}\expandafter\@@IEEEextractgroupremain\expandafter{\ifnum0=`}\fi}

\def\@@IEEEextractgroupremain#1{\def\@IEEEextractedgroupremain{#1}}



% \@IEEEextracttoken relocated at top because margin setting commands rely on it



% usage: \@IEEEextracttokengroups*{}
% \@IEEEextracttokengroups fully expands its argument (which it then stores
% in \@IEEEextracttokengroupsarg) and then assigns the first "brace group" of
% tokens (with the outermost braces removed) to the macro
% \@IEEEextractedfirstgroup.
% The meaning of the first nonbrace (but including the empty group) token
% within this first group is assigned via \let to \@IEEEextractedfirsttoken
% as well as stored in the macro \@IEEEextractedfirsttokenmacro. If a first
% nonbrace token does not exist (or is an empty group), these will be \relax
% and empty, respectively.  Tokens that would otherwise be discarded during
% the acquisition of the first token in the first group are stored in
% \@IEEEextractedfirsttokensdiscarded, however their original relative brace
% nesting depths are not guaranteed to be preserved.
% The first group within this first group is stored in the macro
% \@IEEEextractedfirstfirstgroup.
% Likewise for the next group after the first: \@IEEEextractednextgroup,
% \@IEEEextractednextfirstgroup, \@IEEEextractednextgroupfirsttoken,
% \@IEEEextractednextgroupfirsttokenmacro, and 
% \@IEEEextractednextfirsttokensdiscarded.
% All tokens/groups after the first group, including any enclosing braces,
% are stored in the macro \@IEEEextractedafterfirstgroupremain which will
% be empty if none exist.
%
% For example:
% \@IEEEextracttokengroups{{{ab}{cd}}{{ef}g}}
% will result in:
%
% \@IEEEextracttokengroupsarg             ==> a macro containing {{ab}{cd}}{{ef}g}
% \@IEEEextractedfirstgroup               ==> a macro containing {ab}{cd}
% \@IEEEextractedafterfirstgroupremain    ==> a macro containing {{ef}g}
% \@IEEEextractedfirsttoken               ==> the letter a
% \@IEEEextractedfirsttokenmacro          ==> a macro containing a
% \@IEEEextractedfirsttokensdiscarded     ==> a macro containing bcd
% \@IEEEextractedfirstfirstgroup          ==> a macro containing ab
% \@IEEEextractednextgroup                ==> a macro containing {ef}g
% \@IEEEextractednextfirsttoken           ==> the letter e
% \@IEEEextractednextfirsttokenmacro      ==> a macro containing e
% \@IEEEextractednextfirsttokensdiscarded ==> a macro containing fg
% \@IEEEextractednextfirstgroup           ==> a macro containing ef
%
% If given an empty argument, \@IEEEextractedfirsttoken and
% \@IEEEextractednextfirsttoken will be set to \relax
% and all the macros will be empty.
% the *-star form, \@IEEEextracttokengroups*, does not expand its argument
% contents during processing.
%
% Depends on: \@IEEEextractgroup, \@IEEEextracttoken
\def\@IEEEextracttokengroups{\@ifstar{\let\@IEEEextracttokengroupsdef=\def\@@IEEEextracttokengroups}{\let\@IEEEextracttokengroupsdef=\edef\@@IEEEextracttokengroups}}
\def\@@IEEEextracttokengroups#1{\@IEEEextracttokengroupsdef\@IEEEextracttokengroupsarg{#1}\relax
% begin extraction, these functions are safe with empty arguments
% first group
\expandafter\@IEEEextractgroup\expandafter*\expandafter{\@IEEEextracttokengroupsarg}\relax
\let\@IEEEextractedfirstgroup\@IEEEextractedgroup
\let\@IEEEextractedafterfirstgroupremain\@IEEEextractedgroupremain
\expandafter\@IEEEextracttoken\expandafter*\expandafter{\@IEEEextractedfirstgroup}\relax
\let\@IEEEextractedfirsttoken\@IEEEextractedtoken
\let\@IEEEextractedfirsttokenmacro\@IEEEextractedtokenmacro
\let\@IEEEextractedfirsttokensdiscarded\@IEEEextractedtokensdiscarded
% first first group
\expandafter\@IEEEextractgroup\expandafter*\expandafter{\@IEEEextractedfirstgroup}\relax
\let\@IEEEextractedfirstfirstgroup\@IEEEextractedgroup
% next group
\expandafter\@IEEEextractgroup\expandafter*\expandafter{\@IEEEextractedafterfirstgroupremain}\relax
\let\@IEEEextractednextgroup\@IEEEextractedgroup
\expandafter\@IEEEextracttoken\expandafter*\expandafter{\@IEEEextractednextgroup}\relax
\let\@IEEEextractednextfirsttoken\@IEEEextractedtoken
\let\@IEEEextractednextfirsttokenmacro\@IEEEextractedtokenmacro
\let\@IEEEextractednextfirsttokensdiscarded\@IEEEextractedtokensdiscarded
% next first group
\expandafter\@IEEEextractgroup\expandafter*\expandafter{\@IEEEextractednextgroup}\relax
\let\@IEEEextractednextfirstgroup\@IEEEextractedgroup}


%% -- End of Command Argument Scanning Support Functions --




%%
%% START OF IEEEeqnarray DEFINITIONS
%%
%% Inspired by the concepts, examples, and previous works of LaTeX 
%% coders and developers such as Donald Arseneau, Fred Bartlett, 
%% David Carlisle, Tony Liu, Frank Mittelbach, Piet van Oostrum, 
%% Roland Winkler and Mark Wooding.
%% I don't make the claim that my work here is even near their calibre. ;)


\newif\if@IEEEeqnarrayboxnojot% flag to indicate if the environment was called as the star form
\@IEEEeqnarrayboxnojotfalse

\newif\if@advanceIEEEeqncolcnt% tracks if the environment should advance the col counter
% allows a way to make an \IEEEeqnarraybox that can be used within an \IEEEeqnarray
% used by IEEEeqnarraymulticol so that it can work properly in both
\@advanceIEEEeqncolcnttrue

\newcount\@IEEEeqnnumcols % tracks how many IEEEeqnarray cols are defined
\newcount\@IEEEeqncolcnt  % tracks how many IEEEeqnarray cols the user actually used


% The default math style used by the columns
\def\IEEEeqnarraymathstyle{\displaystyle}
% The default text style used by the columns
% default to using the current font
\def\IEEEeqnarraytextstyle{\relax}

% like the iedlistdecl but for \IEEEeqnarray
\def\IEEEeqnarraydecl{\relax}
\def\IEEEeqnarrayboxdecl{\relax}



% V1.8 flags to indicate that equation numbering is to persist
\newif\if@IEEEeqnumpersist%
\@IEEEeqnumpersistfalse
\newif\if@IEEEsubeqnumpersist%
\@IEEEsubeqnumpersistfalse
%
% V1.8 flags to indicate if (sub)equation number of last line was preadvanced
\newif\if@IEEEeqnumpreadv%
\@IEEEeqnumpreadvfalse
\newif\if@IEEEsubeqnumpreadv%
\@IEEEsubeqnumpreadvfalse

\newcount\@IEEEsubeqnnumrollback% saves previous value of IEEEsubequation number in case we need to restore it

% \yesnumber is the opposite of \nonumber
% a novel concept with the same def as the equationarray package
% However, we give IEEE versions too since some LaTeX packages such as 
% the MDWtools mathenv.sty redefine \nonumber to something else.
% This command is intended for use in non-IEEEeqnarray math environments
\providecommand{\yesnumber}{\global\@eqnswtrue}


% IEEEyes/nonumber 
% V1.8 add persistant * forms
% These commands can alter the type of equation an IEEEeqnarray line is.
\def\IEEEyesnumber{\@ifstar{\global\@IEEEeqnumpersisttrue\global\@IEEEsubeqnumpersistfalse\@IEEEyesnumber}{\@IEEEyesnumber}}

\def\@IEEEyesnumber{\global\@eqnswtrue
\if@IEEEeqnarrayISinner% alter counters and label only inside an IEEEeqnarray
\ifnum\c@IEEEsubequation>0\relax
   \stepcounter{equation}\setcounter{IEEEsubequation}{0}\gdef\@currentlabel{\p@equation\theequation}\relax
   \gdef\@currentHref{\@IEEEtheHrefequation}% setup hyperref label
\fi
% even if we reached this eqn num via a preadv, it is legit now
\global\@IEEEeqnumpreadvfalse\global\@IEEEsubeqnumpreadvfalse
\fi}

\def\IEEEnonumber{\@ifstar{\global\@IEEEeqnumpersistfalse\global\@IEEEsubeqnumpersistfalse\global\@eqnswfalse}{\global\@eqnswfalse}}


\def\IEEEyessubnumber{\@ifstar{\global\@IEEEsubeqnumpersisttrue\@IEEEyessubnumber}{\@IEEEyessubnumber}}
%
\def\@IEEEyessubnumber{\if@IEEEeqnarrayISinner% alter counters and label only inside an IEEEeqnarray
  \ifnum\c@IEEEsubequation>0\relax% if it already is a subequation, we are good to go as-is
  \else% if we are a regular equation we have to watch out for two cases
    \if@IEEEeqnumpreadv% if this equation is the result of a preadvance, backout and bump the sub eqnnum
       \global\advance\c@equation\m@ne\global\c@IEEEsubequation=\@IEEEsubeqnnumrollback\addtocounter{IEEEsubequation}{1}\relax
    \else% non-preadvanced equations just need initialization of their sub eqnnum
       \setcounter{IEEEsubequation}{1}\relax
    \fi
  \fi% fi already is subequation
  \gdef\@currentlabel{\p@IEEEsubequation\theIEEEsubequation}\relax
  \gdef\@currentHref{\@IEEEtheHrefsubequation}% setup hyperref label
  \global\@IEEEeqnumpreadvfalse\global\@IEEEsubeqnumpreadvfalse% no longer a preadv anymore
  \global\@eqnswtrue
\fi}


\def\IEEEnosubnumber{\@ifstar{\global\@IEEEsubeqnumpersistfalse\@IEEEnosubnumber}{\@IEEEnosubnumber}}
%
\def\@IEEEnosubnumber{\if@IEEEeqnarrayISinner% alter counters and label only inside an IEEEeqnarray
  \if@eqnsw % we do nothing unless we know we will display because we play with the counters here
    % if it currently is a subequation, bump up to the next equation number and turn off the subequation
    \ifnum\c@IEEEsubequation>0\relax\addtocounter{equation}{1}\setcounter{IEEEsubequation}{0}\relax
    \fi
    \global\@IEEEeqnumpreadvfalse\global\@IEEEsubeqnumpreadvfalse% no longer a preadv anymore
    \gdef\@currentlabel{\p@equation\theequation}\relax
    \gdef\@currentHref{\@IEEEtheHrefequation}% setup hyperref label
  \fi
\fi}



% allows users to "push away" equations that get too close to the equation numbers
\def\IEEEeqnarraynumspace{\hphantom{\ifnum\c@IEEEsubequation>0\relax\theIEEEsubequationdis\else\theequationdis\fi}}

% provides a way to span multiple columns within IEEEeqnarray environments
% will consider \if@advanceIEEEeqncolcnt before globally advancing the
% column counter - so as to work within \IEEEeqnarraybox
% usage: \IEEEeqnarraymulticol{number cols. to span}{col type}{cell text}
\long\def\IEEEeqnarraymulticol#1#2#3{\multispan{#1}\relax
% check if column is defined for the precolumn definition
% We have to be careful here because TeX scans for & even within an \iffalse
% where it does not expand macros. So, if we used only one \ifx and a #3
% appeared in the false branch and the user inserted another alignment
% structure that uses & in the \IEEEeqnarraymulticol{}, TeX will not see that
% there is an inner alignment in the false branch yet still will see any &
% there and will think that they apply to the outer alignment resulting in an
% incomplete \ifx error.
% So, here we use separate checks for the pre and post parts in order to keep
% the #3 outside of all conditionals.
\relax\expandafter\ifx\csname @IEEEeqnarraycolDEF#2\endcsname\@IEEEeqnarraycolisdefined\relax
\csname @IEEEeqnarraycolPRE#2\endcsname
\else% if not, error and use default type
\@IEEEclspkgerror{Invalid column type "#2" in \string\IEEEeqnarraymulticol.\MessageBreak
Using a default centering column instead}%
{You must define IEEEeqnarray column types before use.}%
\csname @IEEEeqnarraycolPRE@IEEEdefault\endcsname
\fi
% The ten \relax are to help prevent misleading error messages in case a user
% accidently inserted a macro that tries to acquire additional arguments.
#3\relax\relax\relax\relax\relax\relax\relax\relax\relax\relax
% check if column is defined for the postcolumn definition
\expandafter\ifx\csname @IEEEeqnarraycolDEF#2\endcsname\@IEEEeqnarraycolisdefined\relax
\csname @IEEEeqnarraycolPOST#2\endcsname
\else% if not, use the default type
\csname @IEEEeqnarraycolPOST@IEEEdefault\endcsname
\fi
% advance column counter only if the IEEEeqnarray environment wants it
\if@advanceIEEEeqncolcnt\global\advance\@IEEEeqncolcnt by #1\relax\fi}

% like \omit, but maintains track of the column counter for \IEEEeqnarray
\def\IEEEeqnarrayomit{\omit\if@advanceIEEEeqncolcnt\global\advance\@IEEEeqncolcnt by 1\relax\fi}


% provides a way to define a letter referenced column type
% usage: \IEEEeqnarraydefcol{col. type letter/name}{pre insertion text}{post insertion text}
\def\IEEEeqnarraydefcol#1#2#3{\expandafter\def\csname @IEEEeqnarraycolPRE#1\endcsname{#2}%
\expandafter\def\csname @IEEEeqnarraycolPOST#1\endcsname{#3}%
\expandafter\def\csname @IEEEeqnarraycolDEF#1\endcsname{1}}


% provides a way to define a numerically referenced inter-column glue types
% usage: \IEEEeqnarraydefcolsep{col. glue number}{glue definition}
\def\IEEEeqnarraydefcolsep#1#2{\expandafter\def\csname @IEEEeqnarraycolSEP\romannumeral #1\endcsname{#2}%
\expandafter\def\csname @IEEEeqnarraycolSEPDEF\romannumeral #1\endcsname{1}}


\def\@IEEEeqnarraycolisdefined{1}% just a macro for 1, used for checking undefined column types


% expands and appends the given argument to the \@IEEEtrantmptoksA token list
% used to build up the \halign preamble
\def\@IEEEappendtoksA#1{\edef\@@IEEEappendtoksA{\@IEEEtrantmptoksA={\the\@IEEEtrantmptoksA #1}}%
\@@IEEEappendtoksA}

% also appends to \@IEEEtrantmptoksA, but does not expand the argument
% uses \toks8 as a scratchpad register
\def\@IEEEappendNOEXPANDtoksA#1{\toks8={#1}%
\edef\@@IEEEappendNOEXPANDtoksA{\@IEEEtrantmptoksA={\the\@IEEEtrantmptoksA\the\toks8}}%
\@@IEEEappendNOEXPANDtoksA}

% define some common column types for the user
% math
\IEEEeqnarraydefcol{l}{$\IEEEeqnarraymathstyle}{$\hfil}
\IEEEeqnarraydefcol{c}{\hfil$\IEEEeqnarraymathstyle}{$\hfil}
\IEEEeqnarraydefcol{r}{\hfil$\IEEEeqnarraymathstyle}{$}
\IEEEeqnarraydefcol{L}{$\IEEEeqnarraymathstyle{}}{{}$\hfil}
\IEEEeqnarraydefcol{C}{\hfil$\IEEEeqnarraymathstyle{}}{{}$\hfil}
\IEEEeqnarraydefcol{R}{\hfil$\IEEEeqnarraymathstyle{}}{{}$}
% text
\IEEEeqnarraydefcol{s}{\IEEEeqnarraytextstyle}{\hfil}
\IEEEeqnarraydefcol{t}{\hfil\IEEEeqnarraytextstyle}{\hfil}
\IEEEeqnarraydefcol{u}{\hfil\IEEEeqnarraytextstyle}{}

% vertical rules
\IEEEeqnarraydefcol{v}{}{\vrule width\arrayrulewidth}
\IEEEeqnarraydefcol{vv}{\vrule width\arrayrulewidth\hfil}{\hfil\vrule width\arrayrulewidth}
\IEEEeqnarraydefcol{V}{}{\vrule width\arrayrulewidth\hskip\doublerulesep\vrule width\arrayrulewidth}
\IEEEeqnarraydefcol{VV}{\vrule width\arrayrulewidth\hskip\doublerulesep\vrule width\arrayrulewidth\hfil}%
{\hfil\vrule width\arrayrulewidth\hskip\doublerulesep\vrule width\arrayrulewidth}

% horizontal rules
\IEEEeqnarraydefcol{h}{}{\leaders\hrule height\arrayrulewidth\hfil}
\IEEEeqnarraydefcol{H}{}{\leaders\vbox{\hrule width\arrayrulewidth\vskip\doublerulesep\hrule width\arrayrulewidth}\hfil}

% plain
\IEEEeqnarraydefcol{x}{}{}
\IEEEeqnarraydefcol{X}{$}{$}

% the default column type to use in the event a column type is not defined
\IEEEeqnarraydefcol{@IEEEdefault}{\hfil$\IEEEeqnarraymathstyle}{$\hfil}


% a zero tabskip (used for "-" col types)
\def\@IEEEeqnarraycolSEPzero{0pt plus 0pt minus 0pt}
% a centering tabskip (used for "+" col types)
\def\@IEEEeqnarraycolSEPcenter{1000pt plus 0pt minus 1000pt}

% top level default tabskip glues for the start, end, and inter-column
% may be reset within environments not always at the top level, e.g., \IEEEeqnarraybox
\edef\@IEEEeqnarraycolSEPdefaultstart{\@IEEEeqnarraycolSEPcenter}% default start glue
\edef\@IEEEeqnarraycolSEPdefaultend{\@IEEEeqnarraycolSEPcenter}% default end glue
\edef\@IEEEeqnarraycolSEPdefaultmid{\@IEEEeqnarraycolSEPzero}% default inter-column glue



% creates a vertical rule that extends from the bottom to the top a a cell
% Provided in case other packages redefine \vline some other way.
% usage: \IEEEeqnarrayvrule[rule thickness]
% If no argument is provided, \arrayrulewidth will be used for the rule thickness. 
\newcommand\IEEEeqnarrayvrule[1][\arrayrulewidth]{\vrule\@width#1\relax}

% creates a blank separator row
% usage: \IEEEeqnarrayseprow[separation length][font size commands]
% default is \IEEEeqnarrayseprow[0.25\normalbaselineskip][\relax]
% blank arguments inherit the default values
% uses \skip5 as a scratch register - calls \@IEEEeqnarraystrutsize which uses more scratch registers
\def\IEEEeqnarrayseprow{\relax\@ifnextchar[{\@IEEEeqnarrayseprow}{\@IEEEeqnarrayseprow[0.25\normalbaselineskip]}}
\def\@IEEEeqnarrayseprow[#1]{\relax\@ifnextchar[{\@@IEEEeqnarrayseprow[#1]}{\@@IEEEeqnarrayseprow[#1][\relax]}}
\def\@@IEEEeqnarrayseprow[#1][#2]{\def\@IEEEeqnarrayseprowARGONE{#1}%
\ifx\@IEEEeqnarrayseprowARGONE\@empty%
% get the skip value, based on the font commands
% use skip5 because \IEEEeqnarraystrutsize uses \skip0, \skip2, \skip3
% assign within a bogus box to confine the font changes
{\setbox0=\hbox{#2\relax\global\skip5=0.25\normalbaselineskip}}%
\else%
{\setbox0=\hbox{#2\relax\global\skip5=#1}}%
\fi%
\@IEEEeqnarrayhoptolastcolumn\IEEEeqnarraystrutsize{\skip5}{0pt}[\relax]\relax}

% creates a blank separator row, but omits all the column templates
% usage: \IEEEeqnarrayseprowcut[separation length][font size commands]
% default is \IEEEeqnarrayseprowcut[0.25\normalbaselineskip][\relax]
% blank arguments inherit the default values
% uses \skip5 as a scratch register - calls \@IEEEeqnarraystrutsize which uses more scratch registers
\def\IEEEeqnarrayseprowcut{\multispan{\@IEEEeqnnumcols}\relax% span all the cols
% advance column counter only if the IEEEeqnarray environment wants it
\if@advanceIEEEeqncolcnt\global\advance\@IEEEeqncolcnt by \@IEEEeqnnumcols\relax\fi%
\@ifnextchar[{\@IEEEeqnarrayseprowcut}{\@IEEEeqnarrayseprowcut[0.25\normalbaselineskip]}}
\def\@IEEEeqnarrayseprowcut[#1]{\relax\@ifnextchar[{\@@IEEEeqnarrayseprowcut[#1]}{\@@IEEEeqnarrayseprowcut[#1][\relax]}}
\def\@@IEEEeqnarrayseprowcut[#1][#2]{\def\@IEEEeqnarrayseprowARGONE{#1}%
\ifx\@IEEEeqnarrayseprowARGONE\@empty%
% get the skip value, based on the font commands
% use skip5 because \IEEEeqnarraystrutsize uses \skip0, \skip2, \skip3
% assign within a bogus box to confine the font changes
{\setbox0=\hbox{#2\relax\global\skip5=0.25\normalbaselineskip}}%
\else%
{\setbox0=\hbox{#2\relax\global\skip5=#1}}%
\fi%
\IEEEeqnarraystrutsize{\skip5}{0pt}[\relax]\relax}



% draws a single rule across all the columns optional
% argument determines the rule width, \arrayrulewidth is the default
% updates column counter as needed and turns off struts
% usage: \IEEEeqnarrayrulerow[rule line thickness]
\def\IEEEeqnarrayrulerow{\multispan{\@IEEEeqnnumcols}\relax% span all the cols
% advance column counter only if the IEEEeqnarray environment wants it
\if@advanceIEEEeqncolcnt\global\advance\@IEEEeqncolcnt by \@IEEEeqnnumcols\relax\fi%
\@ifnextchar[{\@IEEEeqnarrayrulerow}{\@IEEEeqnarrayrulerow[\arrayrulewidth]}}
\def\@IEEEeqnarrayrulerow[#1]{\leaders\hrule height#1\hfil\relax% put in our rule 
% turn off any struts
\IEEEeqnarraystrutsize{0pt}{0pt}[\relax]\relax}


% draws a double rule by using a single rule row, a separator row, and then
% another single rule row 
% first optional argument determines the rule thicknesses, \arrayrulewidth is the default
% second optional argument determines the rule spacing, \doublerulesep is the default
% usage: \IEEEeqnarraydblrulerow[rule line thickness][rule spacing]
\def\IEEEeqnarraydblrulerow{\multispan{\@IEEEeqnnumcols}\relax% span all the cols
% advance column counter only if the IEEEeqnarray environment wants it
\if@advanceIEEEeqncolcnt\global\advance\@IEEEeqncolcnt by \@IEEEeqnnumcols\relax\fi%
\@ifnextchar[{\@IEEEeqnarraydblrulerow}{\@IEEEeqnarraydblrulerow[\arrayrulewidth]}}
\def\@IEEEeqnarraydblrulerow[#1]{\relax\@ifnextchar[{\@@IEEEeqnarraydblrulerow[#1]}%
{\@@IEEEeqnarraydblrulerow[#1][\doublerulesep]}}
\def\@@IEEEeqnarraydblrulerow[#1][#2]{\def\@IEEEeqnarraydblrulerowARG{#1}%
% we allow the user to say \IEEEeqnarraydblrulerow[][]
\ifx\@IEEEeqnarraydblrulerowARG\@empty%
\@IEEEeqnarrayrulerow[\arrayrulewidth]%
\else%
\@IEEEeqnarrayrulerow[#1]\relax%
\fi%
\def\@IEEEeqnarraydblrulerowARG{#2}%
\ifx\@IEEEeqnarraydblrulerowARG\@empty%
\\\IEEEeqnarrayseprow[\doublerulesep][\relax]%
\else%
\\\IEEEeqnarrayseprow[#2][\relax]%
\fi%
\\\multispan{\@IEEEeqnnumcols}%
% advance column counter only if the IEEEeqnarray environment wants it
\if@advanceIEEEeqncolcnt\global\advance\@IEEEeqncolcnt by \@IEEEeqnnumcols\relax\fi%
\def\@IEEEeqnarraydblrulerowARG{#1}%
\ifx\@IEEEeqnarraydblrulerowARG\@empty%
\@IEEEeqnarrayrulerow[\arrayrulewidth]%
\else%
\@IEEEeqnarrayrulerow[#1]%
\fi%
}

% draws a double rule by using a single rule row, a separator (cutting) row, and then
% another single rule row 
% first optional argument determines the rule thicknesses, \arrayrulewidth is the default
% second optional argument determines the rule spacing, \doublerulesep is the default
% usage: \IEEEeqnarraydblrulerow[rule line thickness][rule spacing]
\def\IEEEeqnarraydblrulerowcut{\multispan{\@IEEEeqnnumcols}\relax% span all the cols
% advance column counter only if the IEEEeqnarray environment wants it
\if@advanceIEEEeqncolcnt\global\advance\@IEEEeqncolcnt by \@IEEEeqnnumcols\relax\fi%
\@ifnextchar[{\@IEEEeqnarraydblrulerowcut}{\@IEEEeqnarraydblrulerowcut[\arrayrulewidth]}}
\def\@IEEEeqnarraydblrulerowcut[#1]{\relax\@ifnextchar[{\@@IEEEeqnarraydblrulerowcut[#1]}%
{\@@IEEEeqnarraydblrulerowcut[#1][\doublerulesep]}}
\def\@@IEEEeqnarraydblrulerowcut[#1][#2]{\def\@IEEEeqnarraydblrulerowARG{#1}%
% we allow the user to say \IEEEeqnarraydblrulerow[][]
\ifx\@IEEEeqnarraydblrulerowARG\@empty%
\@IEEEeqnarrayrulerow[\arrayrulewidth]%
\else%
\@IEEEeqnarrayrulerow[#1]%
\fi%
\def\@IEEEeqnarraydblrulerowARG{#2}%
\ifx\@IEEEeqnarraydblrulerowARG\@empty%
\\\IEEEeqnarrayseprowcut[\doublerulesep][\relax]%
\else%
\\\IEEEeqnarrayseprowcut[#2][\relax]%
\fi%
\\\multispan{\@IEEEeqnnumcols}%
% advance column counter only if the IEEEeqnarray environment wants it
\if@advanceIEEEeqncolcnt\global\advance\@IEEEeqncolcnt by \@IEEEeqnnumcols\relax\fi%
\def\@IEEEeqnarraydblrulerowARG{#1}%
\ifx\@IEEEeqnarraydblrulerowARG\@empty%
\@IEEEeqnarrayrulerow[\arrayrulewidth]%
\else%
\@IEEEeqnarrayrulerow[#1]%
\fi%
}



% inserts a full row's worth of &'s
% relies on \@IEEEeqnnumcols to provide the correct number of columns
% uses \@IEEEtrantmptoksA, \count0 as scratch registers
\def\@IEEEeqnarrayhoptolastcolumn{\@IEEEtrantmptoksA={}\count0=1\relax%
\loop% add cols if the user did not use them all
\ifnum\count0<\@IEEEeqnnumcols\relax%
\@IEEEappendtoksA{&}%
\advance\count0 by 1\relax% update the col count
\repeat%
\the\@IEEEtrantmptoksA%execute the &'s
}



\newif\if@IEEEeqnarrayISinner % flag to indicate if we are within the lines
\@IEEEeqnarrayISinnerfalse    % of an IEEEeqnarray - after the IEEEeqnarraydecl

\edef\@IEEEeqnarrayTHEstrutheight{0pt} % height and depth of IEEEeqnarray struts
\edef\@IEEEeqnarrayTHEstrutdepth{0pt}

\edef\@IEEEeqnarrayTHEmasterstrutheight{0pt} % default height and depth of
\edef\@IEEEeqnarrayTHEmasterstrutdepth{0pt}  % struts within an IEEEeqnarray

\edef\@IEEEeqnarrayTHEmasterstrutHSAVE{0pt} % saved master strut height
\edef\@IEEEeqnarrayTHEmasterstrutDSAVE{0pt} % and depth

\newif\if@IEEEeqnarrayusemasterstrut % flag to indicate that the master strut value
\@IEEEeqnarrayusemasterstruttrue     % is to be used



% saves the strut height and depth of the master strut
\def\@IEEEeqnarraymasterstrutsave{\relax%
\expandafter\skip0=\@IEEEeqnarrayTHEmasterstrutheight\relax%
\expandafter\skip2=\@IEEEeqnarrayTHEmasterstrutdepth\relax%
% remove stretchability
\dimen0\skip0\relax%
\dimen2\skip2\relax%
% save values
\edef\@IEEEeqnarrayTHEmasterstrutHSAVE{\the\dimen0}%
\edef\@IEEEeqnarrayTHEmasterstrutDSAVE{\the\dimen2}}

% restores the strut height and depth of the master strut
\def\@IEEEeqnarraymasterstrutrestore{\relax%
\expandafter\skip0=\@IEEEeqnarrayTHEmasterstrutHSAVE\relax%
\expandafter\skip2=\@IEEEeqnarrayTHEmasterstrutDSAVE\relax%
% remove stretchability
\dimen0\skip0\relax%
\dimen2\skip2\relax%
% restore values
\edef\@IEEEeqnarrayTHEmasterstrutheight{\the\dimen0}%
\edef\@IEEEeqnarrayTHEmasterstrutdepth{\the\dimen2}}


% globally restores the strut height and depth to the 
% master values and sets the master strut flag to true
\def\@IEEEeqnarraystrutreset{\relax%
\expandafter\skip0=\@IEEEeqnarrayTHEmasterstrutheight\relax%
\expandafter\skip2=\@IEEEeqnarrayTHEmasterstrutdepth\relax%
% remove stretchability
\dimen0\skip0\relax%
\dimen2\skip2\relax%
% restore values
\xdef\@IEEEeqnarrayTHEstrutheight{\the\dimen0}%
\xdef\@IEEEeqnarrayTHEstrutdepth{\the\dimen2}%
\global\@IEEEeqnarrayusemasterstruttrue}


% if the master strut is not to be used, make the current
% values of \@IEEEeqnarrayTHEstrutheight, \@IEEEeqnarrayTHEstrutdepth
% and the use master strut flag, global
% this allows user strut commands issued in the last column to be carried
% into the isolation/strut column
\def\@IEEEeqnarrayglobalizestrutstatus{\relax%
\if@IEEEeqnarrayusemasterstrut\else%
\xdef\@IEEEeqnarrayTHEstrutheight{\@IEEEeqnarrayTHEstrutheight}%
\xdef\@IEEEeqnarrayTHEstrutdepth{\@IEEEeqnarrayTHEstrutdepth}%
\global\@IEEEeqnarrayusemasterstrutfalse%
\fi}



% usage: \IEEEeqnarraystrutsize{height}{depth}[font size commands]
% If called outside the lines of an IEEEeqnarray, sets the height
% and depth of both the master and local struts. If called inside
% an IEEEeqnarray line, sets the height and depth of the local strut
% only and sets the flag to indicate the use of the local strut
% values. If the height or depth is left blank, 0.7\normalbaselineskip
% and 0.3\normalbaselineskip will be used, respectively.
% The optional argument can be used to evaluate the lengths under
% a different font size and styles. If none is specified, the current
% font is used.
% uses scratch registers \skip0, \skip2, \skip3, \dimen0, \dimen2
\def\IEEEeqnarraystrutsize#1#2{\relax\@ifnextchar[{\@IEEEeqnarraystrutsize{#1}{#2}}{\@IEEEeqnarraystrutsize{#1}{#2}[\relax]}}
\def\@IEEEeqnarraystrutsize#1#2[#3]{\def\@IEEEeqnarraystrutsizeARG{#1}%
\ifx\@IEEEeqnarraystrutsizeARG\@empty%
{\setbox0=\hbox{#3\relax\global\skip3=0.7\normalbaselineskip}}%
\skip0=\skip3\relax%
\else% arg one present
{\setbox0=\hbox{#3\relax\global\skip3=#1\relax}}%
\skip0=\skip3\relax%
\fi% if null arg
\def\@IEEEeqnarraystrutsizeARG{#2}%
\ifx\@IEEEeqnarraystrutsizeARG\@empty%
{\setbox0=\hbox{#3\relax\global\skip3=0.3\normalbaselineskip}}%
\skip2=\skip3\relax%
\else% arg two present
{\setbox0=\hbox{#3\relax\global\skip3=#2\relax}}%
\skip2=\skip3\relax%
\fi% if null arg
% remove stretchability, just to be safe
\dimen0\skip0\relax%
\dimen2\skip2\relax%
% dimen0 = height, dimen2 = depth
\if@IEEEeqnarrayISinner% inner does not touch master strut size
\edef\@IEEEeqnarrayTHEstrutheight{\the\dimen0}%
\edef\@IEEEeqnarrayTHEstrutdepth{\the\dimen2}%
\@IEEEeqnarrayusemasterstrutfalse% do not use master
\else% outer, have to set master strut too
\edef\@IEEEeqnarrayTHEmasterstrutheight{\the\dimen0}%
\edef\@IEEEeqnarrayTHEmasterstrutdepth{\the\dimen2}%
\edef\@IEEEeqnarrayTHEstrutheight{\the\dimen0}%
\edef\@IEEEeqnarrayTHEstrutdepth{\the\dimen2}%
\@IEEEeqnarrayusemasterstruttrue% use master strut
\fi}


% usage: \IEEEeqnarraystrutsizeadd{added height}{added depth}[font size commands]
% If called outside the lines of an IEEEeqnarray, adds the given height
% and depth to both the master and local struts.
% If called inside an IEEEeqnarray line, adds the given height and depth
% to the local strut only and sets the flag to indicate the use 
% of the local strut values.
% In both cases, if a height or depth is left blank, 0pt is used instead.
% The optional argument can be used to evaluate the lengths under
% a different font size and styles. If none is specified, the current
% font is used.
% uses scratch registers \skip0, \skip2, \skip3, \dimen0, \dimen2
\def\IEEEeqnarraystrutsizeadd#1#2{\relax\@ifnextchar[{\@IEEEeqnarraystrutsizeadd{#1}{#2}}{\@IEEEeqnarraystrutsizeadd{#1}{#2}[\relax]}}
\def\@IEEEeqnarraystrutsizeadd#1#2[#3]{\def\@IEEEeqnarraystrutsizearg{#1}%
\ifx\@IEEEeqnarraystrutsizearg\@empty%
\skip0=0pt\relax%
\else% arg one present
{\setbox0=\hbox{#3\relax\global\skip3=#1}}%
\skip0=\skip3\relax%
\fi% if null arg
\def\@IEEEeqnarraystrutsizearg{#2}%
\ifx\@IEEEeqnarraystrutsizearg\@empty%
\skip2=0pt\relax%
\else% arg two present
{\setbox0=\hbox{#3\relax\global\skip3=#2}}%
\skip2=\skip3\relax%
\fi% if null arg
% remove stretchability, just to be safe
\dimen0\skip0\relax%
\dimen2\skip2\relax%
% dimen0 = height, dimen2 = depth
\if@IEEEeqnarrayISinner% inner does not touch master strut size
% get local strut size
\expandafter\skip0=\@IEEEeqnarrayTHEstrutheight\relax%
\expandafter\skip2=\@IEEEeqnarrayTHEstrutdepth\relax%
% add it to the user supplied values
\advance\dimen0 by \skip0\relax%
\advance\dimen2 by \skip2\relax%
% update the local strut size
\edef\@IEEEeqnarrayTHEstrutheight{\the\dimen0}%
\edef\@IEEEeqnarrayTHEstrutdepth{\the\dimen2}%
\@IEEEeqnarrayusemasterstrutfalse% do not use master
\else% outer, have to set master strut too
% get master strut size
\expandafter\skip0=\@IEEEeqnarrayTHEmasterstrutheight\relax%
\expandafter\skip2=\@IEEEeqnarrayTHEmasterstrutdepth\relax%
% add it to the user supplied values
\advance\dimen0 by \skip0\relax%
\advance\dimen2 by \skip2\relax%
% update the local and master strut sizes
\edef\@IEEEeqnarrayTHEmasterstrutheight{\the\dimen0}%
\edef\@IEEEeqnarrayTHEmasterstrutdepth{\the\dimen2}%
\edef\@IEEEeqnarrayTHEstrutheight{\the\dimen0}%
\edef\@IEEEeqnarrayTHEstrutdepth{\the\dimen2}%
\@IEEEeqnarrayusemasterstruttrue% use master strut
\fi}


% allow user a way to see the struts
\newif\ifIEEEvisiblestruts
\IEEEvisiblestrutsfalse

% inserts an invisible strut using the master or local strut values
% uses scratch registers \skip0, \skip2, \dimen0, \dimen2
\def\@IEEEeqnarrayinsertstrut{\relax%
\if@IEEEeqnarrayusemasterstrut
% get master strut size
\expandafter\skip0=\@IEEEeqnarrayTHEmasterstrutheight\relax%
\expandafter\skip2=\@IEEEeqnarrayTHEmasterstrutdepth\relax%
\else%
% get local strut size
\expandafter\skip0=\@IEEEeqnarrayTHEstrutheight\relax%
\expandafter\skip2=\@IEEEeqnarrayTHEstrutdepth\relax%
\fi%
% remove stretchability, probably not needed
\dimen0\skip0\relax%
\dimen2\skip2\relax%
% dimen0 = height, dimen2 = depth
% allow user to see struts if desired
\ifIEEEvisiblestruts%
\vrule width0.2pt height\dimen0 depth\dimen2\relax%
\else%
\vrule width0pt height\dimen0 depth\dimen2\relax\fi}


% creates an invisible strut, useable even outside \IEEEeqnarray
% if \IEEEvisiblestrutstrue, the strut will be visible and 0.2pt wide. 
% usage: \IEEEstrut[height][depth][font size commands]
% default is \IEEEstrut[0.7\normalbaselineskip][0.3\normalbaselineskip][\relax]
% blank arguments inherit the default values
% uses \dimen0, \dimen2, \skip0, \skip2
\def\IEEEstrut{\relax\@ifnextchar[{\@IEEEstrut}{\@IEEEstrut[0.7\normalbaselineskip]}}
\def\@IEEEstrut[#1]{\relax\@ifnextchar[{\@@IEEEstrut[#1]}{\@@IEEEstrut[#1][0.3\normalbaselineskip]}}
\def\@@IEEEstrut[#1][#2]{\relax\@ifnextchar[{\@@@IEEEstrut[#1][#2]}{\@@@IEEEstrut[#1][#2][\relax]}}
\def\@@@IEEEstrut[#1][#2][#3]{\mbox{#3\relax%
\def\@IEEEstrutARG{#1}%
\ifx\@IEEEstrutARG\@empty%
\skip0=0.7\normalbaselineskip\relax%
\else%
\skip0=#1\relax%
\fi%
\def\@IEEEstrutARG{#2}%
\ifx\@IEEEstrutARG\@empty%
\skip2=0.3\normalbaselineskip\relax%
\else%
\skip2=#2\relax%
\fi%
% remove stretchability, probably not needed
\dimen0\skip0\relax%
\dimen2\skip2\relax%
\ifIEEEvisiblestruts%
\vrule width0.2pt height\dimen0 depth\dimen2\relax%
\else%
\vrule width0.0pt height\dimen0 depth\dimen2\relax\fi}}


% enables strut mode by setting a default strut size and then zeroing the
% \baselineskip, \lineskip, \lineskiplimit and \jot
\def\IEEEeqnarraystrutmode{\IEEEeqnarraystrutsize{0.7\normalbaselineskip}{0.3\normalbaselineskip}[\relax]%
\baselineskip=0pt\lineskip=0pt\lineskiplimit=0pt\jot=0pt}


% equation and subequation forms to use to setup hyperref's \@currentHref
\def\@IEEEtheHrefequation{equation.\theHequation}
\def\@IEEEtheHrefsubequation{equation.\theHequation\alph{IEEEsubequation}}


\def\IEEEeqnarray{\@IEEEeqnumpersisttrue\@IEEEsubeqnumpersistfalse\@IEEEeqnarray}
\def\endIEEEeqnarray{\end@IEEEeqnarray}

\@namedef{IEEEeqnarray*}{\@IEEEeqnumpersistfalse\@IEEEsubeqnumpersistfalse\@IEEEeqnarray}
\@namedef{endIEEEeqnarray*}{\end@IEEEeqnarray}


% \IEEEeqnarray is an enhanced \eqnarray. 
% The star form defaults to not putting equation numbers at the end of each row.
% usage: \IEEEeqnarray[decl]{cols}
\def\@IEEEeqnarray{\relax\@ifnextchar[{\@@IEEEeqnarray}{\@@IEEEeqnarray[\relax]}}
% We have to be careful here to normalize catcodes just before acquiring the
% cols as that specification may contain punctuation which could be subject
% to document catcode changes.
\def\@@IEEEeqnarray[#1]{\begingroup\IEEEnormalcatcodes\@@@IEEEeqnarray[#1]}
\def\@@@IEEEeqnarray[#1]#2{\endgroup
   % default to showing the equation number or not based on whether or not
   % the star form was involked
   \if@IEEEeqnumpersist\global\@eqnswtrue
   \else% not the star form
   \global\@eqnswfalse
   \fi% if star form
   % provide a basic hyperref \theHequation if this has not already been setup (hyperref not loaded, or no section counter)
   \@ifundefined{theHequation}{\def\theHequation{\arabic{equation}}}{}\relax
   % provide dummy hyperref commands in case hyperref is not loaded
   \providecommand{\Hy@raisedlink}[1]{}\relax
   \providecommand{\hyper@anchorstart}[1]{}\relax
   \providecommand{\hyper@anchorend}{}\relax
   \providecommand{\@currentHref}{}\relax
   \@IEEEeqnumpreadvfalse% reset eqnpreadv flag
   \@IEEEsubeqnumpreadvfalse% reset subeqnpreadv flag
   \@IEEEeqnarrayISinnerfalse% not yet within the lines of the halign
   \@IEEEeqnarraystrutsize{0pt}{0pt}[\relax]% turn off struts by default
   \@IEEEeqnarrayusemasterstruttrue% use master strut till user asks otherwise
   \IEEEvisiblestrutsfalse% diagnostic mode defaults to off
   % no extra space unless the user specifically requests it
   \lineskip=0pt\relax
   \lineskiplimit=0pt\relax
   \baselineskip=\normalbaselineskip\relax%
   \jot=\IEEEnormaljot\relax%
   \mathsurround\z@\relax% no extra spacing around math
   \@advanceIEEEeqncolcnttrue% advance the col counter for each col the user uses, 
                             % used in \IEEEeqnarraymulticol and in the preamble build
   %V1.8 Here we preadvance to the next equation number.
   % If the user later wants a continued subequation, we can roll back.
   \global\@IEEEsubeqnnumrollback=\c@IEEEsubequation%
   \stepcounter{equation}\@IEEEeqnumpreadvtrue% advance equation counter before first line
   \setcounter{IEEEsubequation}{0}% no subequation yet
   \let\@IEEEcurrentlabelsave\@currentlabel% save current label as we later change it globally
   \let\@IEEEcurrentHrefsave\@currentHref% save current href label as we later change it globally
   \def\@currentlabel{\p@equation\theequation}% redefine the ref label
   \def\@currentHref{\@IEEEtheHrefequation}% setup hyperref label
   \IEEEeqnarraydecl\relax% allow a way for the user to make global overrides
   #1\relax% allow user to override defaults
   \let\\\@IEEEeqnarraycr% replace newline with one that can put in eqn. numbers
   \global\@IEEEeqncolcnt\z@% col. count = 0 for first line
   \@IEEEbuildpreamble{#2}\relax% build the preamble and put it into \@IEEEtrantmptoksA 
   % put in the column for the equation number
   \ifnum\@IEEEeqnnumcols>0\relax\@IEEEappendtoksA{&}\fi% col separator for those after the first
   \toks0={##}%
   % advance the \@IEEEeqncolcnt for the isolation col, this helps with error checking
   \@IEEEappendtoksA{\global\advance\@IEEEeqncolcnt by 1\relax}%
   % add the isolation column
   \@IEEEappendtoksA{\tabskip\z@skip\bgroup\the\toks0\egroup}%
   % advance the \@IEEEeqncolcnt for the equation number col, this helps with error checking
   \@IEEEappendtoksA{&\global\advance\@IEEEeqncolcnt by 1\relax}%
   % add the equation number col to the preamble
   \@IEEEappendtoksA{\tabskip\z@skip\hb@xt@\z@\bgroup\hss\the\toks0\egroup}%
   % note \@IEEEeqnnumcols does not count the equation col or isolation col
   % set the starting tabskip glue as determined by the preamble build
   \tabskip=\@IEEEBPstartglue\relax
   % begin the display alignment
   \@IEEEeqnarrayISinnertrue% commands are now within the lines
   $$\everycr{}\halign to\displaywidth\bgroup
   % "exspand" the preamble
   \span\the\@IEEEtrantmptoksA\cr}

% enter isolation/strut column (or the next column if the user did not use
% every column), record the strut status, complete the columns, do the strut if needed,
% restore counters (to backout any equation setup for a next line that was never used)
% to their correct values and exit
\def\end@IEEEeqnarray{\@IEEEeqnarrayglobalizestrutstatus&\@@IEEEeqnarraycr\egroup
\if@IEEEsubeqnumpreadv\global\advance\c@IEEEsubequation\m@ne\fi
\if@IEEEeqnumpreadv\global\advance\c@equation\m@ne\global\c@IEEEsubequation=\@IEEEsubeqnnumrollback\fi
\global\let\@currentlabel\@IEEEcurrentlabelsave% restore current label
\global\let\@currentHref\@IEEEcurrentHrefsave% restore current href label
$$\@ignoretrue}


% IEEEeqnarray uses a modifed \\ instead of the plain \cr to
% end rows. This allows for things like \\*[vskip amount]
% These "cr" macros are modified versions of those for LaTeX2e's eqnarray
% the {\ifnum0=`} braces must be kept away from the last column to avoid
% altering spacing of its math, so we use & to advance to the next column
% as there is an isolation/strut column after the user's columns
\def\@IEEEeqnarraycr{\@IEEEeqnarrayglobalizestrutstatus&% save strut status and advance to next column
   {\ifnum0=`}\fi
   \@ifstar{%
      \global\@eqpen\@M\@IEEEeqnarrayYCR
   }{%
      \global\@eqpen\interdisplaylinepenalty \@IEEEeqnarrayYCR
   }%
}

\def\@IEEEeqnarrayYCR{\@testopt\@IEEEeqnarrayXCR\z@skip}

\def\@IEEEeqnarrayXCR[#1]{%
   \ifnum0=`{\fi}%
   \@@IEEEeqnarraycr
   \noalign{\penalty\@eqpen\vskip\jot\vskip #1\relax}}%

\def\@@IEEEeqnarraycr{\@IEEEtrantmptoksA={}% clear token register
    \advance\@IEEEeqncolcnt by -1\relax% adjust col count because of the isolation column
    \ifnum\@IEEEeqncolcnt>\@IEEEeqnnumcols\relax
    \@IEEEclspkgerror{Too many columns within the IEEEeqnarray\MessageBreak
                          environment}%
    {Use fewer \string &'s or put more columns in the IEEEeqnarray column\MessageBreak 
     specifications.}\relax%
    \else
    \loop% add cols if the user did not use them all
    \ifnum\@IEEEeqncolcnt<\@IEEEeqnnumcols\relax
    \@IEEEappendtoksA{&}%
    \advance\@IEEEeqncolcnt by 1\relax% update the col count
    \repeat
    % this number of &'s will take us the the isolation column
    \fi
    % execute the &'s
    \the\@IEEEtrantmptoksA%
    % handle the strut/isolation column
    \@IEEEeqnarrayinsertstrut% do the strut if needed
    \@IEEEeqnarraystrutreset% reset the strut system for next line or IEEEeqnarray
    &% and enter the equation number column
    \if@eqnsw% only if we display something
      \Hy@raisedlink{\hyper@anchorstart{\@currentHref}}% start a hyperref anchor
      \global\@IEEEeqnumpreadvfalse\relax% displaying an equation number means
      \global\@IEEEsubeqnumpreadvfalse\relax% the equation counters point to valid equations
      % V1.8 Here we setup the counters, currentlabel and status for what would be the *next*
      % equation line as would be the case under the current settings. However, there are two problems.
      % One problem is that there might not ever be a next line. The second problem is that the user
      % may later alter the meaning of a line with commands such as \IEEEyessubnumber. So, to handle
      % these cases we have to record the current values of the (sub)equation counters and revert back
      % to them if the next line is changed or never comes. The \if@IEEEeqnumpreadv, \if@IEEEsubeqnumpreadv
      % and \@IEEEsubeqnnumrollback stuff tracks this.
      % The logic to handle all this is surprisingly complex, but a nice feature of the approach here is
      % that the equation counters and labels remain valid for what the line would be unless a
      % \IEEEyessubnumber et al. later changes it. So, any hyperref links are always correct.
      \ifnum\c@IEEEsubequation>0\relax% handle subequation
         \theIEEEsubequationdis\relax
         \if@IEEEsubeqnumpersist% setup for default type of next line
            \stepcounter{IEEEsubequation}\global\@IEEEsubeqnumpreadvtrue\relax
            \gdef\@currentlabel{\p@IEEEsubequation\theIEEEsubequation}\relax
            \gdef\@currentHref{\@IEEEtheHrefsubequation}% setup hyperref label
         \else
             % if no subeqnum persist, go ahead and setup for a new equation number
             \global\@IEEEsubeqnnumrollback=\c@IEEEsubequation
             \stepcounter{equation}\global\@IEEEeqnumpreadvtrue\relax
             \setcounter{IEEEsubequation}{0}\gdef\@currentlabel{\p@equation\theequation}\relax
             \gdef\@currentHref{\@IEEEtheHrefequation}% setup hyperref label
         \fi
      \else% display a standard equation number
        \theequationdis\relax
        \setcounter{IEEEsubequation}{0}\relax% not really needed
        \if@IEEEsubeqnumpersist% setup for default type of next line
           % subequations that follow plain equations carry the same equation number e.g, 5, 5a rather than 5, 6a
           \stepcounter{IEEEsubequation}\global\@IEEEsubeqnumpreadvtrue\relax
           \gdef\@currentlabel{\p@IEEEsubequation\theIEEEsubequation}\relax
           \gdef\@currentHref{\@IEEEtheHrefsubequation}% setup hyperref label
         \else
             % if no subeqnum persist, go ahead and setup for a new equation number
             \global\@IEEEsubeqnnumrollback=\c@IEEEsubequation
             \stepcounter{equation}\global\@IEEEeqnumpreadvtrue\relax
             \setcounter{IEEEsubequation}{0}\gdef\@currentlabel{\p@equation\theequation}\relax
             \gdef\@currentHref{\@IEEEtheHrefequation}% setup hyperref label
         \fi
      \fi%
      \Hy@raisedlink{\hyper@anchorend}% end hyperref anchor
    \fi% fi only if we display something
    % reset the flags to indicate the default preferences of the display of equation numbers
    \if@IEEEeqnumpersist\global\@eqnswtrue\else\global\@eqnswfalse\fi
    \if@IEEEsubeqnumpersist\global\@eqnswtrue\fi% ditto for the subequation flag
    % reset the number of columns the user actually used
    \global\@IEEEeqncolcnt\z@\relax
    % the real end of the line
    \cr}





% \IEEEeqnarraybox is like \IEEEeqnarray except the box form puts everything
% inside a vtop, vbox, or vcenter box depending on the letter in the second
% optional argument (t,b,c). Vbox is the default. Unlike \IEEEeqnarray,
% equation numbers are not displayed and \IEEEeqnarraybox can be nested.
% \IEEEeqnarrayboxm is for math mode (like \array) and does not put the vbox
% within an hbox.
% \IEEEeqnarrayboxt is for text mode (like \tabular) and puts the vbox within
% a \hbox{$ $} construct.
% \IEEEeqnarraybox will auto detect whether to use \IEEEeqnarrayboxm or 
% \IEEEeqnarrayboxt depending on the math mode.
% The third optional argument specifies the width this box is to be set to -
% natural width is the default.
% The * forms do not add \jot line spacing
% usage: \IEEEeqnarraybox[decl][pos][width]{cols}
\def\IEEEeqnarrayboxm{\@IEEEeqnarrayboxnojotfalse\@IEEEeqnarrayboxHBOXSWfalse\@IEEEeqnarraybox}
\def\endIEEEeqnarrayboxm{\end@IEEEeqnarraybox}
\@namedef{IEEEeqnarrayboxm*}{\@IEEEeqnarrayboxnojottrue\@IEEEeqnarrayboxHBOXSWfalse\@IEEEeqnarraybox}
\@namedef{endIEEEeqnarrayboxm*}{\end@IEEEeqnarraybox}

\def\IEEEeqnarrayboxt{\@IEEEeqnarrayboxnojotfalse\@IEEEeqnarrayboxHBOXSWtrue\@IEEEeqnarraybox}
\def\endIEEEeqnarrayboxt{\end@IEEEeqnarraybox}
\@namedef{IEEEeqnarrayboxt*}{\@IEEEeqnarrayboxnojottrue\@IEEEeqnarrayboxHBOXSWtrue\@IEEEeqnarraybox}
\@namedef{endIEEEeqnarrayboxt*}{\end@IEEEeqnarraybox}

\def\IEEEeqnarraybox{\@IEEEeqnarrayboxnojotfalse\ifmmode\@IEEEeqnarrayboxHBOXSWfalse\else\@IEEEeqnarrayboxHBOXSWtrue\fi%
\@IEEEeqnarraybox}
\def\endIEEEeqnarraybox{\end@IEEEeqnarraybox}

\@namedef{IEEEeqnarraybox*}{\@IEEEeqnarrayboxnojottrue\ifmmode\@IEEEeqnarrayboxHBOXSWfalse\else\@IEEEeqnarrayboxHBOXSWtrue\fi%
\@IEEEeqnarraybox}
\@namedef{endIEEEeqnarraybox*}{\end@IEEEeqnarraybox}

% flag to indicate if the \IEEEeqnarraybox needs to put things into an hbox{$ $} 
% for \vcenter in non-math mode
\newif\if@IEEEeqnarrayboxHBOXSW%
\@IEEEeqnarrayboxHBOXSWfalse

\def\@IEEEeqnarraybox{\relax\@ifnextchar[{\@@IEEEeqnarraybox}{\@@IEEEeqnarraybox[\relax]}}
% We have to be careful here to normalize catcodes just before acquiring the
% cols as that specification may contain punctuation which could be subject
% to document catcode changes.
\def\@@IEEEeqnarraybox[#1]{\relax\begingroup\IEEEnormalcatcodes\@ifnextchar[{\@@@IEEEeqnarraybox[#1]}{\@@@IEEEeqnarraybox[#1][b]}}
\def\@@@IEEEeqnarraybox[#1][#2]{\relax\@ifnextchar[{\@@@@IEEEeqnarraybox[#1][#2]}{\@@@@IEEEeqnarraybox[#1][#2][\relax]}}

% #1 = decl; #2 = t,b,c; #3 = width, #4 = col specs
\def\@@@@IEEEeqnarraybox[#1][#2][#3]#4{\endgroup\@IEEEeqnarrayISinnerfalse % not yet within the lines of the halign
   \@IEEEeqnarraymasterstrutsave% save current master strut values
   \@IEEEeqnarraystrutsize{0pt}{0pt}[\relax]% turn off struts by default
   \@IEEEeqnarrayusemasterstruttrue% use master strut till user asks otherwise
   \IEEEvisiblestrutsfalse% diagnostic mode defaults to off
   % no extra space unless the user specifically requests it
   \lineskip=0pt\relax%
   \lineskiplimit=0pt\relax%
   \baselineskip=\normalbaselineskip\relax%
   \jot=\IEEEnormaljot\relax%
   \mathsurround\z@\relax% no extra spacing around math
   % the default end glues are zero for an \IEEEeqnarraybox
   \edef\@IEEEeqnarraycolSEPdefaultstart{\@IEEEeqnarraycolSEPzero}% default start glue
   \edef\@IEEEeqnarraycolSEPdefaultend{\@IEEEeqnarraycolSEPzero}% default end glue
   \edef\@IEEEeqnarraycolSEPdefaultmid{\@IEEEeqnarraycolSEPzero}% default inter-column glue
   \@advanceIEEEeqncolcntfalse% do not advance the col counter for each col the user uses, 
                              % used in \IEEEeqnarraymulticol and in the preamble build
   \IEEEeqnarrayboxdecl\relax% allow a way for the user to make global overrides
   #1\relax% allow user to override defaults
   \let\\\@IEEEeqnarrayboxcr% replace newline with one that allows optional spacing
   \@IEEEbuildpreamble{#4}\relax% build the preamble and put it into \@IEEEtrantmptoksA
   % add an isolation column to the preamble to stop \\'s {} from getting into the last col
   \ifnum\@IEEEeqnnumcols>0\relax\@IEEEappendtoksA{&}\fi% col separator for those after the first
   \toks0={##}%
   % add the isolation column to the preamble
   \@IEEEappendtoksA{\tabskip\z@skip\bgroup\the\toks0\egroup}% 
   % set the starting tabskip glue as determined by the preamble build
   \tabskip=\@IEEEBPstartglue\relax
   % begin the alignment
   \everycr{}%
   % use only the very first token to determine the positioning
   \@IEEEextracttoken{#2}\relax
   \ifx\@IEEEextractedtokensdiscarded\@empty\else
     \typeout{** WARNING: IEEEeqnarraybox position specifiers after the first in `\@IEEEextracttokenarg' ignored (line \the\inputlineno).}\relax
   \fi
   % \@IEEEextractedtoken has the first token, the rest are ignored
   % if we need to put things into and hbox and go into math mode, do so now
   \if@IEEEeqnarrayboxHBOXSW \leavevmode \hbox \bgroup $\fi%
   % use the appropriate vbox type
   \if\@IEEEextractedtoken t\relax\vtop\else\if\@IEEEextractedtoken c\relax%
   \vcenter\else\vbox\fi\fi\bgroup%
   \@IEEEeqnarrayISinnertrue% commands are now within the lines
   \ifx#3\relax\halign\else\halign to #3\relax\fi%
   \bgroup
   % "exspand" the preamble
   \span\the\@IEEEtrantmptoksA\cr}

% carry strut status and enter the isolation/strut column, 
% exit from math mode if needed, and exit
\def\end@IEEEeqnarraybox{\@IEEEeqnarrayglobalizestrutstatus% carry strut status
&% enter isolation/strut column
\@IEEEeqnarrayinsertstrut% do strut if needed
\@IEEEeqnarraymasterstrutrestore% restore the previous master strut values
% reset the strut system for next IEEEeqnarray
% (sets local strut values back to previous master strut values)
\@IEEEeqnarraystrutreset%
% ensure last line, exit from halign, close vbox
\crcr\egroup\egroup%
% exit from math mode and close hbox if needed
\if@IEEEeqnarrayboxHBOXSW $\egroup\fi}



% IEEEeqnarraybox uses a modifed \\ instead of the plain \cr to
% end rows. This allows for things like \\[vskip amount]
% This "cr" macros are modified versions those for LaTeX2e's eqnarray
% For IEEEeqnarraybox, \\* is the same as \\
% the {\ifnum0=`} braces must be kept away from the last column to avoid
% altering spacing of its math, so we use & to advance to the isolation/strut column
% carry strut status into isolation/strut column
\def\@IEEEeqnarrayboxcr{\@IEEEeqnarrayglobalizestrutstatus% carry strut status
&% enter isolation/strut column
\@IEEEeqnarrayinsertstrut% do strut if needed
% reset the strut system for next line or IEEEeqnarray
\@IEEEeqnarraystrutreset%
{\ifnum0=`}\fi%
\@ifstar{\@IEEEeqnarrayboxYCR}{\@IEEEeqnarrayboxYCR}}

% test and setup the optional argument to \\[]
\def\@IEEEeqnarrayboxYCR{\@testopt\@IEEEeqnarrayboxXCR\z@skip}

% IEEEeqnarraybox does not automatically increase line spacing by \jot
\def\@IEEEeqnarrayboxXCR[#1]{\ifnum0=`{\fi}%
\cr\noalign{\if@IEEEeqnarrayboxnojot\else\vskip\jot\fi\vskip#1\relax}}



% usage: \@IEEEbuildpreamble{column specifiers}
% starts the halign preamble build 
% the assembled preamble is put in \@IEEEtrantmptoksA
\def\@IEEEbuildpreamble#1{\@IEEEtrantmptoksA={}% clear token register
\let\@IEEEBPcurtype=u%current column type is not yet known
\let\@IEEEBPprevtype=s%the previous column type was the start
\let\@IEEEBPnexttype=u%next column type is not yet known
% ensure these are valid
\def\@IEEEBPcurglue={0pt plus 0pt minus 0pt}%
\def\@IEEEBPcurcolname{@IEEEdefault}% name of current column definition
% currently acquired numerically referenced glue
% use a name that is easier to remember
\let\@IEEEBPcurnum=\@IEEEtrantmpcountA%
\@IEEEBPcurnum=0%
% tracks number of columns in the preamble
\@IEEEeqnnumcols=0%
% record the default end glues
\edef\@IEEEBPstartglue{\@IEEEeqnarraycolSEPdefaultstart}%
\edef\@IEEEBPendglue{\@IEEEeqnarraycolSEPdefaultend}%
\edef\@IEEEedefMACRO{#1}\relax% fully expand the preamble to support macro containers
% now parse the user's column specifications
% \ignorespaces is used as a delimiter, need at least one trailing \relax because
% \@@IEEEbuildpreamble looks into the future 
\expandafter\@@IEEEbuildpreamble\@IEEEedefMACRO\ignorespaces\relax\relax}


% usage: \@@IEEEbuildpreamble{current column}{next column}
% parses and builds the halign preamble
\def\@@IEEEbuildpreamble#1#2{\let\@@nextIEEEbuildpreamble=\@@IEEEbuildpreamble%
% use only the very first token to check the end
\@IEEEextracttokengroups{#1}\relax
\ifx\@IEEEextractedfirsttoken\ignorespaces\let\@@nextIEEEbuildpreamble=\@@IEEEfinishpreamble\else%
% identify current and next token type
\@IEEEgetcoltype{#1}{\@IEEEBPcurtype}{1}% current, error on invalid
\@IEEEgetcoltype{#2}{\@IEEEBPnexttype}{0}% next, no error on invalid next
% if curtype is a glue, get the glue def
\if\@IEEEBPcurtype g\@IEEEgetcurglue{#1}{\@IEEEBPcurglue}\fi%
% if curtype is a column, get the column def and set the current column name
\if\@IEEEBPcurtype c\@IEEEgetcurcol{#1}\fi%
% if curtype is a numeral, acquire the user defined glue
\if\@IEEEBPcurtype n\@IEEEprocessNcol{#1}\fi%
% process the acquired glue 
\if\@IEEEBPcurtype g\@IEEEprocessGcol\fi%
% process the acquired col 
\if\@IEEEBPcurtype c\@IEEEprocessCcol\fi%
% ready prevtype for next col spec.
\let\@IEEEBPprevtype=\@IEEEBPcurtype%
% be sure and put back the future token(s) as a group
\fi\@@nextIEEEbuildpreamble{#2}}


% usage: \@@IEEEfinishpreamble{discarded}
% executed just after preamble build is completed
% warn about zero cols, and if prevtype type = u, put in end tabskip glue
% argument is not used
\def\@@IEEEfinishpreamble#1{\ifnum\@IEEEeqnnumcols<1\relax
\@IEEEclspkgerror{No column specifiers declared for IEEEeqnarray}%
{At least one column type must be declared for each IEEEeqnarray.}%
\fi%num cols less than 1
%if last type undefined, set default end tabskip glue
\if\@IEEEBPprevtype u\@IEEEappendtoksA{\tabskip=\@IEEEBPendglue}\fi}


% usage: \@IEEEgetcoltype{col specifier}{\output}{error more}
% Identify and return the column specifier's type code in the given
% \output macro:
% n = number
% g = glue (any other char in catagory 12)
% c = letter
% e = \ignorespaces (end of sequence)
% u = undefined
% error mode: 0 = no error message, 1 = error on invalid char
\def\@IEEEgetcoltype#1#2#3{%
% use only the very first token to determine the type
\@IEEEextracttoken{#1}\relax
% \@IEEEextractedtoken has the first token, the rest are discarded
\let#2=u\relax% assume invalid until know otherwise
\ifx\@IEEEextractedtoken\ignorespaces\let#2=e\else
\ifcat\@IEEEextractedtoken\relax\else% screen out control sequences
\if0\@IEEEextractedtoken\let#2=n\else
\if1\@IEEEextractedtoken\let#2=n\else
\if2\@IEEEextractedtoken\let#2=n\else
\if3\@IEEEextractedtoken\let#2=n\else
\if4\@IEEEextractedtoken\let#2=n\else
\if5\@IEEEextractedtoken\let#2=n\else
\if6\@IEEEextractedtoken\let#2=n\else
\if7\@IEEEextractedtoken\let#2=n\else
\if8\@IEEEextractedtoken\let#2=n\else
\if9\@IEEEextractedtoken\let#2=n\else
\ifcat,\@IEEEextractedtoken\let#2=g\relax
\else\ifcat a\@IEEEextractedtoken\let#2=c\relax\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi
\if#2u\relax
\if0\noexpand#3\relax\else\@IEEEclspkgerror{Invalid character in column specifications}%
{Only letters, numerals and certain other symbols are allowed \MessageBreak
as IEEEeqnarray column specifiers.}\fi\fi}


% usage: \@IEEEgetcurcol{col specifier}
% verify the letter referenced column exists
% and return its name in \@IEEEBPcurcolname
% if column specifier is invalid, use the default column @IEEEdefault
\def\@IEEEgetcurcol#1{\expandafter\ifx\csname @IEEEeqnarraycolDEF#1\endcsname\@IEEEeqnarraycolisdefined%
\def\@IEEEBPcurcolname{#1}\else% invalid column name
\@IEEEclspkgerror{Invalid column type "#1" in column specifications.\MessageBreak
Using a default centering column instead}%
{You must define IEEEeqnarray column types before use.}%
\def\@IEEEBPcurcolname{@IEEEdefault}\fi}


% usage: \@IEEEgetcurglue{glue specifier}{\output}
% identify the predefined (punctuation) glue value
% and return it in the given output macro
\def\@IEEEgetcurglue#1#2{%
% ! = \! (neg small)  -0.16667em (-3/18 em)
% , = \, (small)       0.16667em ( 3/18 em)
% : = \: (med)         0.22222em ( 4/18 em)
% ; = \; (large)       0.27778em ( 5/18 em)
% ' = \quad            1em
% " = \qquad           2em
% . = 0.5\arraycolsep
% / = \arraycolsep
% ? = 2\arraycolsep
% * = 1fil
% + = \@IEEEeqnarraycolSEPcenter
% - = \@IEEEeqnarraycolSEPzero
% Note that all em values are referenced to the math font (textfont2) fontdimen6
% value for 1em.
% 
% use only the very first token to determine the type
\@IEEEextracttoken{#1}\relax
\ifx\@IEEEextractedtokensdiscarded\@empty\else
  \typeout{** WARNING: IEEEeqnarray predefined inter-column glue type specifiers after the first in `\@IEEEextracttokenarg' ignored (line \the\inputlineno).}\relax
\fi
% get the math font 1em value
% LaTeX2e's NFSS2 does not preload the fonts, but \IEEEeqnarray needs
% to gain access to the math (\textfont2) font's spacing parameters.
% So we create a bogus box here that uses the math font to ensure
% that \textfont2 is loaded and ready. If this is not done,
% the \textfont2 stuff here may not work.
% Thanks to Bernd Raichle for his 1997 post on this topic.
{\setbox0=\hbox{$\displaystyle\relax$}}%
% fontdimen6 has the width of 1em (a quad).
\@IEEEtrantmpdimenA=\fontdimen6\textfont2\relax%
% identify the glue value based on the first token
% we discard anything after the first
\if!\@IEEEextractedtoken\@IEEEtrantmpdimenA=-0.16667\@IEEEtrantmpdimenA\edef#2{\the\@IEEEtrantmpdimenA}\else
\if,\@IEEEextractedtoken\@IEEEtrantmpdimenA=0.16667\@IEEEtrantmpdimenA\edef#2{\the\@IEEEtrantmpdimenA}\else
\if:\@IEEEextractedtoken\@IEEEtrantmpdimenA=0.22222\@IEEEtrantmpdimenA\edef#2{\the\@IEEEtrantmpdimenA}\else
\if;\@IEEEextractedtoken\@IEEEtrantmpdimenA=0.27778\@IEEEtrantmpdimenA\edef#2{\the\@IEEEtrantmpdimenA}\else
\if'\@IEEEextractedtoken\@IEEEtrantmpdimenA=1\@IEEEtrantmpdimenA\edef#2{\the\@IEEEtrantmpdimenA}\else
\if"\@IEEEextractedtoken\@IEEEtrantmpdimenA=2\@IEEEtrantmpdimenA\edef#2{\the\@IEEEtrantmpdimenA}\else
\if.\@IEEEextractedtoken\@IEEEtrantmpdimenA=0.5\arraycolsep\edef#2{\the\@IEEEtrantmpdimenA}\else
\if/\@IEEEextractedtoken\edef#2{\the\arraycolsep}\else
\if?\@IEEEextractedtoken\@IEEEtrantmpdimenA=2\arraycolsep\edef#2{\the\@IEEEtrantmpdimenA}\else
\if *\@IEEEextractedtoken\edef#2{0pt plus 1fil minus 0pt}\else
\if+\@IEEEextractedtoken\edef#2{\@IEEEeqnarraycolSEPcenter}\else
\if-\@IEEEextractedtoken\edef#2{\@IEEEeqnarraycolSEPzero}\else
\edef#2{\@IEEEeqnarraycolSEPzero}%
\@IEEEclspkgerror{Invalid predefined inter-column glue type "#1" in\MessageBreak
column specifications. Using a default value of\MessageBreak
0pt instead}%
{Only !,:;'"./?*+ and - are valid predefined glue types in the\MessageBreak 
IEEEeqnarray column specifications.}\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi\fi}


% usage: \@IEEEprocessNcol{digit}
% process a numerical digit from the column specification
% and look up the corresponding user defined glue value
% can transform current type from n to g or a as the user defined glue is acquired
\def\@IEEEprocessNcol#1{\if\@IEEEBPprevtype g%
\@IEEEclspkgerror{Back-to-back inter-column glue specifiers in column\MessageBreak
specifications. Ignoring consecutive glue specifiers\MessageBreak
after the first}%
{You cannot have two or more glue types next to each other\MessageBreak 
in the IEEEeqnarray column specifications.}%
\let\@IEEEBPcurtype=a% abort this glue, future digits will be discarded
\@IEEEBPcurnum=0\relax%
\else% if we previously aborted a glue
\if\@IEEEBPprevtype a\@IEEEBPcurnum=0\let\@IEEEBPcurtype=a%maintain digit abortion
\else%acquire this number
% save the previous type before the numerical digits started
\if\@IEEEBPprevtype n\else\let\@IEEEBPprevsavedtype=\@IEEEBPprevtype\fi%
\multiply\@IEEEBPcurnum by 10\relax%
\advance\@IEEEBPcurnum by #1\relax% add in number, \relax is needed to stop TeX's number scan
\if\@IEEEBPnexttype n\else%close acquisition
\expandafter\ifx\csname @IEEEeqnarraycolSEPDEF\expandafter\romannumeral\number\@IEEEBPcurnum\endcsname\@IEEEeqnarraycolisdefined%
\edef\@IEEEBPcurglue{\csname @IEEEeqnarraycolSEP\expandafter\romannumeral\number\@IEEEBPcurnum\endcsname}%
\else%user glue not defined
\@IEEEclspkgerror{Invalid user defined inter-column glue type "\number\@IEEEBPcurnum" in\MessageBreak
column specifications. Using a default value of\MessageBreak
0pt instead}%
{You must define all IEEEeqnarray numerical inter-column glue types via\MessageBreak
\string\IEEEeqnarraydefcolsep \space before they are used in column specifications.}%
\edef\@IEEEBPcurglue{\@IEEEeqnarraycolSEPzero}%
\fi% glue defined or not
\let\@IEEEBPcurtype=g% change the type to reflect the acquired glue
\let\@IEEEBPprevtype=\@IEEEBPprevsavedtype% restore the prev type before this number glue
\@IEEEBPcurnum=0\relax%ready for next acquisition
\fi%close acquisition, get glue
\fi%discard or acquire number
\fi%prevtype glue or not
}


% process an acquired glue
% add any acquired column/glue pair to the preamble
\def\@IEEEprocessGcol{\if\@IEEEBPprevtype a\let\@IEEEBPcurtype=a%maintain previous glue abortions
\else
% if this is the start glue, save it, but do nothing else 
% as this is not used in the preamble, but before
\if\@IEEEBPprevtype s\edef\@IEEEBPstartglue{\@IEEEBPcurglue}%
\else%not the start glue
\if\@IEEEBPprevtype g%ignore if back to back glues
\@IEEEclspkgerror{Back-to-back inter-column glue specifiers in column\MessageBreak
specifications. Ignoring consecutive glue specifiers\MessageBreak
after the first}%
{You cannot have two or more glue types next to each other\MessageBreak 
in the IEEEeqnarray column specifications.}%
\let\@IEEEBPcurtype=a% abort this glue
\else% not a back to back glue
\if\@IEEEBPprevtype c\relax% if the previoustype was a col, add column/glue pair to preamble
\ifnum\@IEEEeqnnumcols>0\relax\@IEEEappendtoksA{&}\fi
\toks0={##}%
% make preamble advance col counter if this environment needs this
\if@advanceIEEEeqncolcnt\@IEEEappendtoksA{\global\advance\@IEEEeqncolcnt by 1\relax}\fi
% insert the column defintion into the preamble, being careful not to expand
% the column definition
\@IEEEappendtoksA{\tabskip=\@IEEEBPcurglue}%
\@IEEEappendNOEXPANDtoksA{\begingroup\csname @IEEEeqnarraycolPRE}%
\@IEEEappendtoksA{\@IEEEBPcurcolname}%
\@IEEEappendNOEXPANDtoksA{\endcsname}%
\@IEEEappendtoksA{\the\toks0}%
\@IEEEappendNOEXPANDtoksA{\relax\relax\relax\relax\relax%
\relax\relax\relax\relax\relax\csname @IEEEeqnarraycolPOST}%
\@IEEEappendtoksA{\@IEEEBPcurcolname}%
\@IEEEappendNOEXPANDtoksA{\endcsname\relax\relax\relax\relax\relax%
\relax\relax\relax\relax\relax\endgroup}%
\advance\@IEEEeqnnumcols by 1\relax%one more column in the preamble
\else% error: non-start glue with no pending column
\@IEEEclspkgerror{Inter-column glue specifier without a prior column\MessageBreak
type in the column specifications. Ignoring this glue\MessageBreak 
specifier}%
{Except for the first and last positions, glue can be placed only\MessageBreak
between column types.}%
\let\@IEEEBPcurtype=a% abort this glue
\fi% previous was a column
\fi% back-to-back glues
\fi% is start column glue
\fi% prev type not a
}


% process an acquired letter referenced column and, if necessary, add it to the preamble
\def\@IEEEprocessCcol{\if\@IEEEBPnexttype g\else
\if\@IEEEBPnexttype n\else
% we have a column followed by something other than a glue (or numeral glue)
% so we must add this column to the preamble now
\ifnum\@IEEEeqnnumcols>0\relax\@IEEEappendtoksA{&}\fi%col separator for those after the first
\if\@IEEEBPnexttype e\@IEEEappendtoksA{\tabskip=\@IEEEBPendglue\relax}\else%put in end glue
\@IEEEappendtoksA{\tabskip=\@IEEEeqnarraycolSEPdefaultmid\relax}\fi% or default mid glue
\toks0={##}%
% make preamble advance col counter if this environment needs this
\if@advanceIEEEeqncolcnt\@IEEEappendtoksA{\global\advance\@IEEEeqncolcnt by 1\relax}\fi
% insert the column definition into the preamble, being careful not to expand
% the column definition
\@IEEEappendNOEXPANDtoksA{\begingroup\csname @IEEEeqnarraycolPRE}%
\@IEEEappendtoksA{\@IEEEBPcurcolname}%
\@IEEEappendNOEXPANDtoksA{\endcsname}%
\@IEEEappendtoksA{\the\toks0}%
\@IEEEappendNOEXPANDtoksA{\relax\relax\relax\relax\relax%
\relax\relax\relax\relax\relax\csname @IEEEeqnarraycolPOST}%
\@IEEEappendtoksA{\@IEEEBPcurcolname}%
\@IEEEappendNOEXPANDtoksA{\endcsname\relax\relax\relax\relax\relax%
\relax\relax\relax\relax\relax\endgroup}%
\advance\@IEEEeqnnumcols by 1\relax%one more column in the preamble
\fi%next type not numeral
\fi%next type not glue
}


%%
%% END OF IEEEeqnarray DEFINITIONS
%%





% set up the running headers and footers
%
% header and footer font and size specifications
\def\@IEEEheaderstyle{\normalfont\scriptsize}
\def\@IEEEfooterstyle{\normalfont\scriptsize}
%
% compsoc uses sans-serif headers and footers
\ifCLASSOPTIONcompsoc
  \def\@IEEEheaderstyle{\normalfont\sffamily\scriptsize}
  \def\@IEEEfooterstyle{\normalfont\sffamily\scriptsize}
\fi


% standard page style, ps@headings
\def\ps@headings{% default to standard twoside headers, no footers
% will change later if the mode requires otherwise
\def\@oddhead{\hbox{}\@IEEEheaderstyle\rightmark\hfil\thepage}\relax
\def\@evenhead{\@IEEEheaderstyle\thepage\hfil\leftmark\hbox{}}\relax
\let\@oddfoot\@empty
\let\@evenfoot\@empty
\ifCLASSOPTIONtechnote
  % technote twoside
  \def\@oddhead{\hbox{}\@IEEEheaderstyle\leftmark\hfil\thepage}\relax
  \def\@evenhead{\@IEEEheaderstyle\thepage\hfil\leftmark\hbox{}}\relax
\fi
\ifCLASSOPTIONdraftcls
  % draft footers
  \def\@oddfoot{\@IEEEfooterstyle\@date\hfil DRAFT}\relax
  \def\@evenfoot{\@IEEEfooterstyle DRAFT\hfil\@date}\relax
\fi
% oneside
\if@twoside\else
  % standard one side headers
  \def\@oddhead{\hbox{}\@IEEEheaderstyle\leftmark\hfil\thepage}\relax
  \let\@evenhead\@empty
  \ifCLASSOPTIONdraftcls
    % oneside draft footers
    \def\@oddfoot{\@IEEEfooterstyle\@date\hfil DRAFT}\relax
    \let\@evenfoot\@empty
  \fi
\fi
% turn off headers for conferences
\ifCLASSOPTIONconference
  \let\@oddhead\@empty
  \let\@evenhead\@empty
\fi
% turn off footers for draftclsnofoot
\ifCLASSOPTIONdraftclsnofoot
  \let\@oddfoot\@empty
  \let\@evenfoot\@empty
\fi}


% title page style, ps@IEEEtitlepagestyle
\def\ps@IEEEtitlepagestyle{% default title page headers, no footers
\def\@oddhead{\hbox{}\@IEEEheaderstyle\leftmark\hfil\thepage}\relax
\def\@evenhead{\@IEEEheaderstyle\thepage\hfil\leftmark\hbox{}}\relax
\let\@oddfoot\@empty
\let\@evenfoot\@empty
% will change later if the mode requires otherwise
\ifCLASSOPTIONdraftcls
  % draft footers
  \ifCLASSOPTIONdraftclsnofoot\else
    % but only if not draftclsnofoot
    \def\@oddfoot{\@IEEEfooterstyle\@date\hfil DRAFT}\relax
    \def\@evenfoot{\@IEEEfooterstyle DRAFT\hfil\@date}\relax
  \fi
\else
  % all nondraft mode footers
  \if@IEEEusingpubid
    % for title pages that are using a pubid
    % do not repeat pubid on the title page if using a peer review cover page
    \ifCLASSOPTIONpeerreview\else
      % for noncompsoc papers, the pubid uses footnotesize and
      % is at the same vertical position as where the last baseline would normally be
      \def\@oddfoot{\hbox{}\hss\@IEEEfooterstyle\footnotesize\raisebox{\footskip}[0pt][0pt]{\@IEEEpubid}\hss\hbox{}}\relax
      \def\@evenfoot{\hbox{}\hss\@IEEEfooterstyle\footnotesize\raisebox{\footskip}[0pt][0pt]{\@IEEEpubid}\hss\hbox{}}\relax
      \ifCLASSOPTIONcompsoc
        % for compsoc papers, the pubid is at the same vertical position as the normal footer
        \def\@oddfoot{\hbox{}\hss\@IEEEfooterstyle\raisebox{0pt}[0pt][0pt]{\@IEEEpubid}\hss\hbox{}}\relax
        \def\@evenfoot{\hbox{}\hss\@IEEEfooterstyle\raisebox{0pt}[0pt][0pt]{\@IEEEpubid}\hss\hbox{}}\relax
      \fi
    \fi
  \fi
\fi
% turn off headers for conferences
\ifCLASSOPTIONconference
  \let\@oddhead\@empty
  \let\@evenhead\@empty
\fi}


% peer review cover page style, ps@IEEEpeerreviewcoverpagestyle
\def\ps@IEEEpeerreviewcoverpagestyle{% default peer review cover no headers, no footers
\let\@oddhead\@empty
\let\@evenhead\@empty
\let\@oddfoot\@empty
\let\@evenfoot\@empty
% will change later if the mode requires otherwise
\ifCLASSOPTIONdraftcls
  % draft footers
  \ifCLASSOPTIONdraftclsnofoot\else
    % but only if not draftclsnofoot
    \def\@oddfoot{\@IEEEfooterstyle\@date\hfil DRAFT}\relax
    \def\@evenfoot{\@IEEEfooterstyle DRAFT\hfil\@date}\relax
  \fi
\else
  % all nondraft mode footers
  \if@IEEEusingpubid
    % for peer review cover pages that are using a pubid
    % for noncompsoc papers, the pubid uses footnotesize and
    % is at the same vertical position as where the last baseline would normally be
    \def\@oddfoot{\hbox{}\hss\@IEEEfooterstyle\footnotesize\raisebox{\footskip}[0pt][0pt]{\@IEEEpubid}\hss\hbox{}}\relax
    \def\@evenfoot{\hbox{}\hss\@IEEEfooterstyle\footnotesize\raisebox{\footskip}[0pt][0pt]{\@IEEEpubid}\hss\hbox{}}\relax
    \ifCLASSOPTIONcompsoc
      % for compsoc papers, the pubid is at the same vertical position as the normal footer
      \def\@oddfoot{\hbox{}\hss\@IEEEfooterstyle\raisebox{0pt}[0pt][0pt]{\@IEEEpubid}\hss\hbox{}}\relax
      \def\@evenfoot{\hbox{}\hss\@IEEEfooterstyle\raisebox{0pt}[0pt][0pt]{\@IEEEpubid}\hss\hbox{}}\relax
    \fi
  \fi
\fi}



%% Defines the command for putting the header.
%% Note that all the text is forced into uppercase, if you have some text
%% that needs to be in lower case, for instance et. al., then either manually
%% set \leftmark and \rightmark or use \MakeLowercase{et. al.} within the
%% arguments to \markboth.
%% V1.7b add \protect to work with Babel
\def\markboth#1#2{\def\leftmark{\MakeUppercase{\protect#1}}%
\def\rightmark{\MakeUppercase{\protect#2}}}

\def\today{\ifcase\month\or
    January\or February\or March\or April\or May\or June\or
    July\or August\or September\or October\or November\or December\fi
    \space\number\day, \number\year}




%% CITATION AND BIBLIOGRAPHY COMMANDS
%% 
%% V1.6 no longer supports the older, nonstandard \shortcite and \citename setup stuff
% 
% 
% Modify Latex2e \@citex to separate citations with "], ["
\def\@citex[#1]#2{%
  \let\@citea\@empty
  \@cite{\@for\@citeb:=#2\do
    {\@citea\def\@citea{], [}%
     \edef\@citeb{\expandafter\@firstofone\@citeb\@empty}%
     \if@filesw\immediate\write\@auxout{\string\citation{\@citeb}}\fi
     \@ifundefined{b@\@citeb}{\mbox{\reset@font\bfseries ?}%
       \G@refundefinedtrue
       \@latex@warning
         {Citation `\@citeb' on page \thepage \space undefined}}%
       {\hbox{\csname b@\@citeb\endcsname}}}}{#1}}

% V1.6 we create hooks for the optional use of Donald Arseneau's
% cite.sty package. cite.sty is "smart" and will notice that the
% following format controls are already defined and will not
% redefine them. The result will be the proper sorting of the
% citation numbers and auto detection of 3 or more entry "ranges" -
% all in IEEE style:  [1], [2], [5]--[7], [12]
% This also allows for an optional note, i.e., \cite[mynote]{..}.
% If the \cite with note has more than one reference, the note will
% be applied to the last of the listed references. It is generally
% desired that if a note is given, only one reference is listed in
% that \cite.
% Thanks to Mr. Arseneau for providing the required format arguments
% to produce the IEEE style.
\def\citepunct{], [}
\def\citedash{]--[}

% V1.7 default to using same font for urls made by url.sty
\AtBeginDocument{\csname url@samestyle\endcsname}

% V1.6 class files should always provide these
\def\newblock{\hskip .11em\@plus.33em\@minus.07em}
\let\@openbib@code\@empty
% V1.8b article.cls is now providing these too
% we do not use \@mkboth, nor alter the page style
\newenvironment{theindex}
               {\if@twocolumn
                  \@restonecolfalse
                \else
                  \@restonecoltrue
                \fi
                \twocolumn[\section*{\indexname}]%
                \parindent\z@
                \parskip\z@ \@plus .3\p@\relax
                \columnseprule \z@
                \columnsep 35\p@
                \let\item\@idxitem}
               {\if@restonecol\onecolumn\else\clearpage\fi}
\newcommand\@idxitem{\par\hangindent 40\p@}
\newcommand\subitem{\@idxitem \hspace*{20\p@}}
\newcommand\subsubitem{\@idxitem \hspace*{30\p@}}
\newcommand\indexspace{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}



% Provide support for the control entries of IEEEtran.bst V1.00 and later.
% V1.7 optional argument allows for a different aux file to be specified in
% order to handle multiple bibliographies. For example, with multibib.sty:
% \newcites{sec}{Secondary Literature}
% \bstctlcite[@auxoutsec]{BSTcontrolhak}
\def\bstctlcite{\@ifnextchar[{\@bstctlcite}{\@bstctlcite[@auxout]}}
\def\@bstctlcite[#1]#2{\@bsphack
  \@for\@citeb:=#2\do{%
    \edef\@citeb{\expandafter\@firstofone\@citeb}%
    \if@filesw\immediate\write\csname #1\endcsname{\string\citation{\@citeb}}\fi}%
  \@esphack}

% \IEEEnoauxwrite{} allows for citations that do not add to or affect
% the order of the existing citation list. Can be useful for \cite
% within \thanks{}.
\DeclareRobustCommand{\IEEEnoauxwrite}[1]{\relax
\if@filesw
\@fileswfalse
#1\relax\relax\relax\relax\relax
\@fileswtrue
\else
#1\relax\relax\relax\relax\relax
\fi}

% V1.6 provide a way for a user to execute a command just before 
% a given reference number - used to insert a \newpage to balance
% the columns on the last page
\edef\@IEEEtriggerrefnum{0}   % the default of zero means that
                              % the command is not executed
\def\@IEEEtriggercmd{\newpage}

% allow the user to alter the triggered command
\long\def\IEEEtriggercmd#1{\long\def\@IEEEtriggercmd{#1}}

% allow user a way to specify the reference number just before the
% command is executed
\def\IEEEtriggeratref#1{\@IEEEtrantmpcountA=#1%
\edef\@IEEEtriggerrefnum{\the\@IEEEtrantmpcountA}}%

% trigger command at the given reference
\def\@IEEEbibitemprefix{\@IEEEtrantmpcountA=\@IEEEtriggerrefnum\relax%
\advance\@IEEEtrantmpcountA by -1\relax%
\ifnum\c@enumiv=\@IEEEtrantmpcountA\relax\@IEEEtriggercmd\relax\fi}


\def\@biblabel#1{[#1]}

% compsoc journals and conferences left align the reference numbers
\@IEEEcompsoconly{\def\@biblabel#1{[#1]\hfill}}

% controls bib item spacing
\def\IEEEbibitemsep{0pt plus .5pt}

\@IEEEcompsocconfonly{\def\IEEEbibitemsep{0.5\baselineskip plus 0.25\baselineskip minus 0.25\baselineskip}}


\def\thebibliography#1{\section*{\refname}%
    \addcontentsline{toc}{section}{\refname}%
    % V1.6 add some rubber space here and provide a command trigger
    \footnotesize\vskip 0.3\baselineskip plus 0.1\baselineskip minus 0.1\baselineskip%
    \list{\@biblabel{\@arabic\c@enumiv}}%
    {\settowidth\labelwidth{\@biblabel{#1}}%
    \leftmargin\labelwidth
    \advance\leftmargin\labelsep\relax
    \itemsep \IEEEbibitemsep\relax
    \usecounter{enumiv}%
    \let\p@enumiv\@empty
    \renewcommand\theenumiv{\@arabic\c@enumiv}}%
    \let\@IEEElatexbibitem\bibitem%
    \def\bibitem{\@IEEEbibitemprefix\@IEEElatexbibitem}%
\def\newblock{\hskip .11em plus .33em minus .07em}%
% originally:
%   \sloppy\clubpenalty4000\widowpenalty4000%
% by adding the \interlinepenalty here, we make it more
% difficult, but not impossible, for LaTeX to break within a reference.
% The IEEE almost never breaks a reference (but they do it more often with
% technotes). You may get an underfull vbox warning around the bibliography, 
% but the final result will be much more like what the IEEE will publish. 
% MDS 11/2000
\ifCLASSOPTIONtechnote\sloppy\clubpenalty4000\widowpenalty4000\interlinepenalty100%
\else\sloppy\clubpenalty4000\widowpenalty4000\interlinepenalty500\fi%
    \sfcode`\.=1000\relax}
\let\endthebibliography=\endlist




% TITLE PAGE COMMANDS
% 
% 
% \IEEEmembership is used to produce the sublargesize italic font used to indicate author 
% IEEE membership. compsoc uses a large size sans slant font
\def\IEEEmembership#1{{\@IEEEnotcompsoconly{\sublargesize}\normalfont\@IEEEcompsoconly{\sffamily}\textit{#1}}}
 

% \IEEEauthorrefmark{} produces a footnote type symbol to indicate author affiliation.
% When given an argument of 1 to 9, \IEEEauthorrefmark{} follows the standard LaTeX footnote
% symbol sequence convention. However, for arguments 10 and above, \IEEEauthorrefmark{} 
% reverts to using lower case roman numerals, so it cannot overflow. Do note that you 
% cannot use \footnotemark[] in place of \IEEEauthorrefmark{} within \author as the footnote
% symbols will have been turned off to prevent \thanks from creating footnote marks.
% \IEEEauthorrefmark{} produces a symbol that appears to LaTeX as having zero vertical
% height - this allows for a more compact line packing, but the user must ensure that
% the interline spacing is large enough to prevent \IEEEauthorrefmark{} from colliding
% with the text above.
% V1.7 make this a robust command
% V1.8 transmag uses an arabic author affiliation symbol
\ifCLASSOPTIONtransmag
\DeclareRobustCommand*{\IEEEauthorrefmark}[1]{\raisebox{0pt}[0pt][0pt]{\textsuperscript{\footnotesize #1}}}
\else
\DeclareRobustCommand*{\IEEEauthorrefmark}[1]{\raisebox{0pt}[0pt][0pt]{\textsuperscript{\footnotesize\ensuremath{\ifcase#1\or *\or \dagger\or \ddagger\or%
    \mathsection\or \mathparagraph\or \|\or **\or \dagger\dagger%
    \or \ddagger\ddagger \else\textsuperscript{\expandafter\romannumeral#1}\fi}}}}
\fi


% FONT CONTROLS AND SPACINGS FOR CONFERENCE MODE AUTHOR NAME AND AFFILIATION BLOCKS
% 
% The default font styles for the author name and affiliation blocks (confmode)
\def\@IEEEauthorblockNstyle{\normalfont\@IEEEcompsocnotconfonly{\sffamily}\sublargesize}
\def\@IEEEauthorblockAstyle{\normalfont\@IEEEcompsocnotconfonly{\sffamily}\@IEEEcompsocconfonly{\itshape}\normalsize}
% The default if the user does not use an author block
\def\@IEEEauthordefaulttextstyle{\normalfont\@IEEEcompsocnotconfonly{\sffamily}\sublargesize}

% adjustment spacing from title (or special paper notice) to author name blocks (confmode)
% can be negative
\def\@IEEEauthorblockconfadjspace{-0.25em}
% compsoc conferences need more space here
\@IEEEcompsocconfonly{\def\@IEEEauthorblockconfadjspace{0.75\@IEEEnormalsizeunitybaselineskip}}

% spacing between name and affiliation blocks (confmode)
% This can be negative.
% The IEEE doesn't want any added spacing here, but I will leave these
% controls in place in case they ever change their mind.
% Personally, I like 0.75ex.
%\def\@IEEEauthorblockNtopspace{0.75ex}
%\def\@IEEEauthorblockAtopspace{0.75ex}
\def\@IEEEauthorblockNtopspace{0.0ex}
\def\@IEEEauthorblockAtopspace{0.0ex}
\ifCLASSOPTIONtransmag
% transmag uses one line of space above first affiliation block
\def\@IEEEauthorblockAtopspace{1\@IEEEnormalsizeunitybaselineskip}
\fi

% baseline spacing within name and affiliation blocks (confmode)
% must be positive, spacings below certain values will make 
% the position of line of text sensitive to the contents of the
% line above it i.e., whether or not the prior line has descenders, 
% subscripts, etc. For this reason it is a good idea to keep
% these above 2.6ex
\def\@IEEEauthorblockNinterlinespace{2.6ex}
\def\@IEEEauthorblockAinterlinespace{2.75ex}

% This tracks the required strut size.
% See the \@IEEEauthorhalign command for the actual default value used.
\def\@IEEEauthorblockXinterlinespace{2.7ex}

% variables to retain font size and style across groups
% values given here have no effect as they will be overwritten later
\gdef\@IEEESAVESTATEfontsize{10}
\gdef\@IEEESAVESTATEfontbaselineskip{12}
\gdef\@IEEESAVESTATEfontencoding{OT1}
\gdef\@IEEESAVESTATEfontfamily{ptm}
\gdef\@IEEESAVESTATEfontseries{m}
\gdef\@IEEESAVESTATEfontshape{n}

% saves the current font attributes
\def\@IEEEcurfontSAVE{\global\let\@IEEESAVESTATEfontsize\f@size%
\global\let\@IEEESAVESTATEfontbaselineskip\f@baselineskip%
\global\let\@IEEESAVESTATEfontencoding\f@encoding%
\global\let\@IEEESAVESTATEfontfamily\f@family%
\global\let\@IEEESAVESTATEfontseries\f@series%
\global\let\@IEEESAVESTATEfontshape\f@shape}

% restores the saved font attributes
\def\@IEEEcurfontRESTORE{\fontsize{\@IEEESAVESTATEfontsize}{\@IEEESAVESTATEfontbaselineskip}%
\fontencoding{\@IEEESAVESTATEfontencoding}%
\fontfamily{\@IEEESAVESTATEfontfamily}%
\fontseries{\@IEEESAVESTATEfontseries}%
\fontshape{\@IEEESAVESTATEfontshape}%
\selectfont}


% variable to indicate if the current block is the first block in the column
\newif\if@IEEEprevauthorblockincol   \@IEEEprevauthorblockincolfalse


% the command places a strut with height and depth = \@IEEEauthorblockXinterlinespace
% we use this technique to have complete manual control over the spacing of the lines
% within the halign environment.
% We set the below baseline portion at 30%, the above
% baseline portion at 70% of the total length.
% Responds to changes in the document's \baselinestretch
\def\@IEEEauthorstrutrule{\@IEEEtrantmpdimenA\@IEEEauthorblockXinterlinespace%
\@IEEEtrantmpdimenA=\baselinestretch\@IEEEtrantmpdimenA%
\rule[-0.3\@IEEEtrantmpdimenA]{0pt}{\@IEEEtrantmpdimenA}}


% blocks to hold the authors' names and affilations. 
% Makes formatting easy for conferences
%
% use real definitions in conference mode
% name block
\def\IEEEauthorblockN#1{\relax\@IEEEauthorblockNstyle% set the default text style
\gdef\@IEEEauthorblockXinterlinespace{0pt}% disable strut for spacer row
% the \expandafter hides the \cr in conditional tex, see the array.sty docs
% for details, probably not needed here as the \cr is in a macro
% do a spacer row if needed
\if@IEEEprevauthorblockincol\expandafter\@IEEEauthorblockNtopspaceline\fi
\global\@IEEEprevauthorblockincoltrue% we now have a block in this column
%restore the correct strut value
\gdef\@IEEEauthorblockXinterlinespace{\@IEEEauthorblockNinterlinespace}%
% input the author names
#1%
% end the row if the user did not already
\crcr}
% spacer row for names
\def\@IEEEauthorblockNtopspaceline{\cr\noalign{\vskip\@IEEEauthorblockNtopspace}}
%
% affiliation block
\def\IEEEauthorblockA#1{\relax\@IEEEauthorblockAstyle% set the default text style
\gdef\@IEEEauthorblockXinterlinespace{0pt}%disable strut for spacer row
% the \expandafter hides the \cr in conditional tex, see the array.sty docs
% for details, probably not needed here as the \cr is in a macro
% do a spacer row if needed
\if@IEEEprevauthorblockincol\expandafter\@IEEEauthorblockAtopspaceline\fi
\global\@IEEEprevauthorblockincoltrue% we now have a block in this column
%restore the correct strut value
\gdef\@IEEEauthorblockXinterlinespace{\@IEEEauthorblockAinterlinespace}%
% input the author affiliations
#1%
% end the row if the user did not already
\crcr
% V1.8 transmag does not use any additional affiliation spacing after the first author
\ifCLASSOPTIONtransmag\gdef\@IEEEauthorblockAtopspace{0pt}\fi}

% spacer row for affiliations
\def\@IEEEauthorblockAtopspaceline{\cr\noalign{\vskip\@IEEEauthorblockAtopspace}}


% allow papers to compile even if author blocks are used in modes other
% than conference or peerreviewca. For such cases, we provide dummy blocks.
\ifCLASSOPTIONconference
\else
   \ifCLASSOPTIONpeerreviewca\else
      % not conference, peerreviewca or transmag mode
      \ifCLASSOPTIONtransmag\else
         \def\IEEEauthorblockN#1{#1}%
         \def\IEEEauthorblockA#1{#1}%
      \fi
   \fi
\fi



% we provide our own halign so as not to have to depend on tabular
\def\@IEEEauthorhalign{\@IEEEauthordefaulttextstyle% default text style
   \lineskip=0pt\relax% disable line spacing
   \lineskiplimit=0pt\relax%
   \baselineskip=0pt\relax%
   \@IEEEcurfontSAVE% save the current font
   \mathsurround\z@\relax% no extra spacing around math
   \let\\\@IEEEauthorhaligncr% replace newline with halign friendly one
   \tabskip=0pt\relax% no column spacing
   \everycr{}% ensure no problems here
   \@IEEEprevauthorblockincolfalse% no author blocks yet
   \def\@IEEEauthorblockXinterlinespace{2.7ex}% default interline space
   \vtop\bgroup%vtop box
   \halign\bgroup&\relax\hfil\@IEEEcurfontRESTORE\relax ##\relax
   \hfil\@IEEEcurfontSAVE\@IEEEauthorstrutrule\cr}

% ensure last line, exit from halign, close vbox
\def\end@IEEEauthorhalign{\crcr\egroup\egroup}

% handle bogus star form
\def\@IEEEauthorhaligncr{{\ifnum0=`}\fi\@ifstar{\@@IEEEauthorhaligncr}{\@@IEEEauthorhaligncr}}

% test and setup the optional argument to \\[]
\def\@@IEEEauthorhaligncr{\@testopt\@@@IEEEauthorhaligncr\z@skip}

% end the line and do the optional spacer
\def\@@@IEEEauthorhaligncr[#1]{\ifnum0=`{\fi}\cr\noalign{\vskip#1\relax}}



% flag to prevent multiple \and warning messages
\newif\if@IEEEWARNand
\@IEEEWARNandtrue

% if in conference or peerreviewca modes, we support the use of \and as \author is a
% tabular environment, otherwise we warn the user that \and is invalid
% outside of conference or peerreviewca modes.
\def\and{\relax} % provide a bogus \and that we will then override

\renewcommand{\and}[1][\relax]{\if@IEEEWARNand\typeout{** WARNING: \noexpand\and is valid only
                               when in conference or peerreviewca}\typeout{modes (line \the\inputlineno).}\fi\global\@IEEEWARNandfalse}

\ifCLASSOPTIONconference%
\renewcommand{\and}[1][\hfill]{\end{@IEEEauthorhalign}#1\begin{@IEEEauthorhalign}}%
\fi
\ifCLASSOPTIONpeerreviewca
\renewcommand{\and}[1][\hfill]{\end{@IEEEauthorhalign}#1\begin{@IEEEauthorhalign}}%
\fi
% V1.8 transmag uses conference author format
\ifCLASSOPTIONtransmag
\renewcommand{\and}[1][\hfill]{\end{@IEEEauthorhalign}#1\begin{@IEEEauthorhalign}}%
\fi

% page clearing command
% based on LaTeX2e's \cleardoublepage, but allows different page styles
% for the inserted blank pages
\def\@IEEEcleardoublepage#1{\clearpage\if@twoside\ifodd\c@page\else
\hbox{}\thispagestyle{#1}\newpage\if@twocolumn\hbox{}\thispagestyle{#1}\newpage\fi\fi\fi}

% V1.8b hooks to allow adjustment of space above title
\def\IEEEtitletopspace{0.5\baselineskip}
% an added extra amount to allow for adjustment/offset
\def\IEEEtitletopspaceextra{0pt}

% user command to invoke the title page
\def\maketitle{\par%
  \begingroup%
  \normalfont%
  \def\thefootnote{}%  the \thanks{} mark type is empty
  \def\footnotemark{}% and kill space from \thanks within author
  \let\@makefnmark\relax% V1.7, must *really* kill footnotemark to remove all \textsuperscript spacing as well.
  \footnotesize%       equal spacing between thanks lines
  \footnotesep 0.7\baselineskip%see global setting of \footnotesep for more info
  % V1.7 disable \thanks note indention for compsoc
  \@IEEEcompsoconly{\long\def\@makefntext##1{\parindent 1em\noindent\hbox{\@makefnmark}##1}}%
  \normalsize%
  \ifCLASSOPTIONpeerreview
     \newpage\global\@topnum\z@ \@maketitle\@IEEEstatictitlevskip\@IEEEaftertitletext%
     \thispagestyle{IEEEpeerreviewcoverpagestyle}\@thanks%
  \else
     \if@twocolumn%
        \ifCLASSOPTIONtechnote%
           \newpage\global\@topnum\z@ \@maketitle\@IEEEstatictitlevskip\@IEEEaftertitletext%
        \else
           \twocolumn[{\IEEEquantizevspace{\@maketitle}[\IEEEquantizedisabletitlecmds]{0pt}[-\topskip]{\baselineskip}{\@IEEENORMtitlevspace}{\@IEEEMINtitlevspace}\@IEEEaftertitletext}]%
        \fi
     \else
        \newpage\global\@topnum\z@ \@maketitle\@IEEEstatictitlevskip\@IEEEaftertitletext%
     \fi
     \thispagestyle{IEEEtitlepagestyle}\@thanks%
  \fi
  % pullup page for pubid if used.
  \if@IEEEusingpubid
     \enlargethispage{-\@IEEEpubidpullup}%
  \fi 
  \endgroup
  \setcounter{footnote}{0}\let\maketitle\relax\let\@maketitle\relax
  \gdef\@thanks{}%
  % v1.6b do not clear these as we will need the title again for peer review papers
  % \gdef\@author{}\gdef\@title{}%
  \let\thanks\relax}


% V1.8 parbox to format \@IEEEtitleabstractindextext
\long\def\@IEEEtitleabstractindextextbox#1{\parbox{1\textwidth}{#1}}
% V1.8 compsoc is partial width
\ifCLASSOPTIONcompsoc
% comparison with proofs suggests it's in the range of 92.1-92.3%
\long\def\@IEEEtitleabstractindextextbox#1{\parbox{0.922\textwidth}{\@IEEEcompsocnotconfonly{\rightskip\@flushglue\leftskip\z@skip}#1}}
\fi

% formats the Title, authors names, affiliations and special paper notice
% THIS IS A CONTROLLED SPACING COMMAND! Do not allow blank lines or unintentional
% spaces to enter the definition - use % at the end of each line
\def\@maketitle{\newpage
\bgroup\par\vskip\IEEEtitletopspace\vskip\IEEEtitletopspaceextra\centering%
\ifCLASSOPTIONtechnote% technotes, V1.8a abstract and index terms are not treated differently for compsoc technotes
   {\bfseries\large\@IEEEcompsoconly{\Large\sffamily}\@title\par}\vskip 1.3em{\lineskip .5em\@IEEEcompsoconly{\large\sffamily}\@author
   \@IEEEspecialpapernotice\par}\relax
\else% not a technote
   \vskip0.2em{\Huge\ifCLASSOPTIONtransmag\bfseries\LARGE\fi\@IEEEcompsoconly{\sffamily}\@IEEEcompsocconfonly{\normalfont\normalsize\vskip 2\@IEEEnormalsizeunitybaselineskip
   \bfseries\Large}\@IEEEcompsocnotconfonly{\vskip 0.75\@IEEEnormalsizeunitybaselineskip}\@title\par}\relax
   \@IEEEcompsocnotconfonly{\vskip 0.5\@IEEEnormalsizeunitybaselineskip}\vskip1.0em\par%
   % V1.6 handle \author differently if in conference mode
   \ifCLASSOPTIONconference%
      {\@IEEEspecialpapernotice\mbox{}\vskip\@IEEEauthorblockconfadjspace%
       \mbox{}\hfill\begin{@IEEEauthorhalign}\@author\end{@IEEEauthorhalign}\hfill\mbox{}\par}\relax
   \else% peerreviewca, peerreview or journal
      \ifCLASSOPTIONpeerreviewca
         % peerreviewca handles author names just like conference mode
         {\@IEEEcompsoconly{\sffamily}\@IEEEspecialpapernotice\mbox{}\vskip\@IEEEauthorblockconfadjspace%
          \mbox{}\hfill\begin{@IEEEauthorhalign}\@author\end{@IEEEauthorhalign}\hfill\mbox{}\par
          {\@IEEEcompsoconly{\vskip 1.5em\relax
           \@IEEEtitleabstractindextextbox{\@IEEEtitleabstractindextext}\par\noindent\hfill
           \IEEEcompsocdiamondline\hfill\hbox{}\par}}}\relax
      \else% journal, peerreview or transmag
         \ifCLASSOPTIONtransmag
            % transmag also handles author names just like conference mode
            % it also uses \@IEEEtitleabstractindextex, but with one line less
            % space above, and one more below
           {\@IEEEspecialpapernotice\mbox{}\vskip\@IEEEauthorblockconfadjspace%
            \mbox{}\hfill\begin{@IEEEauthorhalign}\@author\end{@IEEEauthorhalign}\hfill\mbox{}\par
           {\vspace{0.5\baselineskip}\relax\@IEEEtitleabstractindextextbox{\@IEEEtitleabstractindextext}\vspace{-1\baselineskip}\par}}\relax
         \else% journal or peerreview
           {\lineskip.5em\@IEEEcompsoconly{\sffamily}\sublargesize\@author\@IEEEspecialpapernotice\par
           {\@IEEEcompsoconly{\vskip 1.5em\relax
            \@IEEEtitleabstractindextextbox{\@IEEEtitleabstractindextext}\par\noindent\hfill
            \IEEEcompsocdiamondline\hfill\hbox{}\par}}}\relax
         \fi
      \fi
   \fi
\fi\par\addvspace{0.5\baselineskip}\egroup}


% V1.7 Computer Society "diamond line" which follows index terms for nonconference papers
% V1.8a full width diamond line for single column use
\def\@IEEEcompsocdiamondlinei{\vrule depth 0pt height 0.5pt width 4cm\nobreak\hspace{7.5pt}\nobreak
\raisebox{-3.5pt}{\fontfamily{pzd}\fontencoding{U}\fontseries{m}\fontshape{n}\fontsize{11}{12}\selectfont\char70}\nobreak
\hspace{7.5pt}\nobreak\vrule depth 0pt height 0.5pt width 4cm\relax}
% V1.8a narrower width diamond line for double column use
\def\@IEEEcompsocdiamondlineii{\vrule depth 0pt height 0.5pt width 2.5cm\nobreak\hspace{7.5pt}\nobreak
\raisebox{-3.5pt}{\fontfamily{pzd}\fontencoding{U}\fontseries{m}\fontshape{n}\fontsize{11}{12}\selectfont\char70}\nobreak
\hspace{7.5pt}\nobreak\vrule depth 0pt height 0.5pt width 2.5cm\relax}
% V1.8a bare core without rules to base a last resort on for very narrow linewidths
\def\@IEEEcompsocdiamondlineiii{\mbox{}\nobreak\hspace{7.5pt}\nobreak
\raisebox{-3.5pt}{\fontfamily{pzd}\fontencoding{U}\fontseries{m}\fontshape{n}\fontsize{11}{12}\selectfont\char70}\nobreak
\hspace{7.5pt}\nobreak\mbox{}\relax}

% V1.8a allow \IEEEcompsocdiamondline to adjust for different linewidths.
% Use \@IEEEcompsocdiamondlinei if its width is less than 0.66\linewidth (0.487 nominal for single column)
% if not, fall back to \@IEEEcompsocdiamondlineii if its width is less than 0.75\linewidth (0.659 nominal for double column)
% if all else fails, try to make a custom diamondline based on the abnormally narrow linewidth
\def\IEEEcompsocdiamondline{\settowidth{\@IEEEtrantmpdimenA}{\@IEEEcompsocdiamondlinei}\relax
\ifdim\@IEEEtrantmpdimenA<0.66\linewidth\relax\@IEEEcompsocdiamondlinei\relax
\else
\settowidth{\@IEEEtrantmpdimenA}{\@IEEEcompsocdiamondlineii}\relax
\ifdim\@IEEEtrantmpdimenA<0.75\linewidth\relax\@IEEEcompsocdiamondlineii\relax
\else
\settowidth{\@IEEEtrantmpdimenA}{\@IEEEcompsocdiamondlineiii}\relax
\@IEEEtrantmpdimenB=\linewidth\relax
\addtolength{\@IEEEtrantmpdimenB}{-1\@IEEEtrantmpdimenA}\relax
\vrule depth 0pt height 0.5pt width 0.33\@IEEEtrantmpdimenB\@IEEEcompsocdiamondlineiii\vrule depth 0pt height 0.5pt width 0.33\@IEEEtrantmpdimenB\relax
\fi\fi}


% V1.7 standard LateX2e \thanks, but with \itshape under compsoc. Also make it a \long\def
% We also need to trigger the one-shot footnote rule
\def\@IEEEtriggeroneshotfootnoterule{\global\@IEEEenableoneshotfootnoteruletrue}


\long\def\thanks#1{\footnotemark
    \protected@xdef\@thanks{\@thanks
        \protect\footnotetext[\the\c@footnote]{\@IEEEcompsoconly{\itshape
        \protect\@IEEEtriggeroneshotfootnoterule\relax}\ignorespaces#1}}}
\let\@thanks\@empty


% V1.7 allow \author to contain \par's. This is needed to allow \thanks to contain \par.
\long\def\author#1{\gdef\@author{#1}}


% in addition to setting up IEEEitemize, we need to remove a baselineskip space above and
% below it because \list's \pars introduce blank lines because of the footnote struts.
\def\@IEEEsetupcompsocitemizelist{\def\labelitemi{$\bullet$}%
\setlength{\IEEElabelindent}{0pt}\setlength{\labelsep}{1.2em}\setlength{\parskip}{0pt}%
\setlength{\partopsep}{0pt}\setlength{\topsep}{0.5\baselineskip}\vspace{-1\baselineskip}\relax}


% flag for fake non-compsoc \IEEEcompsocthanksitem - prevents line break on very first item
\newif\if@IEEEbreakcompsocthanksitem \@IEEEbreakcompsocthanksitemfalse

\ifCLASSOPTIONcompsoc
% V1.7 compsoc bullet item \thanks
% also, we need to redefine this to destroy the argument in \IEEEquantizevspace
\long\def\IEEEcompsocitemizethanks#1{\relax\@IEEEbreakcompsocthanksitemfalse\footnotemark
    \protected@xdef\@thanks{\@thanks
        \protect\footnotetext[\the\c@footnote]{\itshape\protect\@IEEEtriggeroneshotfootnoterule
        {\let\IEEEiedlistdecl\relax\protect\begin{IEEEitemize}[\protect\@IEEEsetupcompsocitemizelist]\ignorespaces#1\relax
        \protect\end{IEEEitemize}}\protect\vspace{-1\baselineskip}}}}
\DeclareRobustCommand*{\IEEEcompsocthanksitem}{\item}
\else
% non-compsoc, allow for dual compilation via rerouting to normal \thanks
\long\def\IEEEcompsocitemizethanks#1{\thanks{#1}}
% redirect to "pseudo-par" \hfil\break\indent after swallowing [] from \IEEEcompsocthanksitem[]
\DeclareRobustCommand{\IEEEcompsocthanksitem}{\@ifnextchar [{\@IEEEthanksswallowoptionalarg}%
{\@IEEEthanksswallowoptionalarg[\relax]}}
% be sure and break only after first item, be sure and ignore spaces after optional argument
\def\@IEEEthanksswallowoptionalarg[#1]{\relax\if@IEEEbreakcompsocthanksitem\hfil\break
\indent\fi\@IEEEbreakcompsocthanksitemtrue\ignorespaces}
\fi


% V1.6b define the \IEEEpeerreviewmaketitle as needed
\ifCLASSOPTIONpeerreview
\def\IEEEpeerreviewmaketitle{\@IEEEcleardoublepage{empty}%
\ifCLASSOPTIONtwocolumn
\twocolumn[{\IEEEquantizevspace{\@IEEEpeerreviewmaketitle}[\IEEEquantizedisabletitlecmds]{0pt}[-\topskip]{\baselineskip}{\@IEEENORMtitlevspace}{\@IEEEMINtitlevspace}}]
\else
\newpage\@IEEEpeerreviewmaketitle\@IEEEstatictitlevskip
\fi
\thispagestyle{IEEEtitlepagestyle}}
\else
% \IEEEpeerreviewmaketitle does nothing if peer review option has not been selected
\def\IEEEpeerreviewmaketitle{\relax}
\fi

% peerreview formats the repeated title like the title in journal papers.
\def\@IEEEpeerreviewmaketitle{\bgroup\par\addvspace{0.5\baselineskip}\centering\@IEEEcompsoconly{\sffamily}%
\normalfont\normalsize\vskip0.2em{\Huge\@title\par}\vskip1.0em\par
\par\addvspace{0.5\baselineskip}\egroup}



% V1.6 
% this is a static rubber spacer between the title/authors and the main text
% used for single column text, or when the title appears in the first column
% of two column text (technotes). 
\def\@IEEEstatictitlevskip{{\normalfont\normalsize
% adjust spacing to next text
% v1.6b handle peer review papers
\ifCLASSOPTIONpeerreview
% for peer review papers, the same value is used for both title pages
% regardless of the other paper modes
   \vskip 1\baselineskip plus 0.375\baselineskip minus 0.1875\baselineskip
\else
   \ifCLASSOPTIONconference% conference
      \vskip 1\baselineskip plus 0.375\baselineskip minus 0.1875\baselineskip%
   \else%
      \ifCLASSOPTIONtechnote% technote
         \vskip 1\baselineskip plus 0.375\baselineskip minus 0.1875\baselineskip%
      \else% journal uses more space
         \vskip 2.5\baselineskip plus 0.75\baselineskip minus 0.375\baselineskip%
      \fi
   \fi
\fi}}


% set the nominal and minimum values for the quantized title spacer
% the quantization algorithm will not allow the spacer size to
% become less than \@IEEEMINtitlevspace - instead it will be lengthened
% default to journal values
\def\@IEEENORMtitlevspace{2.5\baselineskip}
\def\@IEEEMINtitlevspace{2\baselineskip}
% conferences and technotes need tighter spacing
\ifCLASSOPTIONconference% conference
  \def\@IEEENORMtitlevspace{1\baselineskip}
  \def\@IEEEMINtitlevspace{0.75\baselineskip}
\fi
\ifCLASSOPTIONtechnote% technote
  \def\@IEEENORMtitlevspace{1\baselineskip}
  \def\@IEEEMINtitlevspace{0.75\baselineskip}
\fi


% V1.8a
\def\IEEEquantizevspace{\begingroup\@ifstar{\@IEEEquantizevspacestarformtrue\@IEEEquantizevspace}{\@IEEEquantizevspacestarformfalse\@IEEEquantizevspace}}
% \IEEEquantizevspace[output dimen register]{object}[object decl]
%                    {top baselineskip}
%                    [offset][prevdepth][lineskip limit][lineskip]
%                    {unit height}{nominal vspace}{minimum vspace}
%
% Calculates and creates the vspace needed to make the combined height with
% the given object an integer multiple of the given unit height. This command
% is more general than the older \@IEEEdynamictitlevspace it replaces.
%
% The star form has no effect at present, but is reserved for future use.
%
% If the optional argument [output dimen register] is given, the calculated
% vspace height is stored in the given output dimen (or skip) register
% and no other action is taken, otherwise the object followed by a vspace*
% of the appropriate height is evaluated/output.
%
% The optional object decl (declarations) is code that is evaluated just
% before the object's height is evaluated. Its intented purpose is to allow
% for the alteration or disabling of code within the object during internal
% height evaluation (e.g., \long\def\thanks#1{\relax} ).
% This special code is not invoked if/when the object is rendered at the end.
%
% The nominal vspace is the target value of the added vspace and the minimum
% vspace is the lower allowed limit. The vspacer will be the value that achieves
% integral overall height, in terms of the given unit height, that is closest
% to the nominal vspace and that is not less than the specified minimum vspace.
%
% The line spacing algorithm of TeX is somewhat involved and requires special
% care with regard to the first line of a vertical list (which is indicated
% when \prevdepth is -1000pt or less). top baselineskip specifies the
% baselineskip or topskip used prior to the object. If the height of the
% first line of the object is greater than the given top baselineskip, then
% the top baselineskip is subtracted from the height of the first line and
% that difference is considered along with the rest of the object height
% (because the object will be shifted down by an amount = 
% top line height - top baselineskip). Otherwise, the height of the first line
% of the object is ignored as far as the calculations are concerned.
% This algorithm is adequate for objects that appear at the top of a page
% (e.g., titles) where \topskip spacing is used.
%
% However, as explained on page 78 of the TeXbook, interline spacing is more
% complex when \baselineskip is being used (indicated by \prevdepth > 
% -1000pt). The four optional parameters offset, prevdepth, lineskip limit and
% lineskip are assumed to be equal to be 0pt, \prevdepth, \lineskiplimit and
% \lineskip, respectively, if they are omitted.
%
% The prevdepth is the depth of the line before the object, the lineskip limit
% specifies how close the top of the object can come to the bottom of the
% previous line before \baselineskip is ignored and \lineskip is inserted
% between the object and the line above it. Lineskip does not come into
% play unless the first line of the object is high enough to "get too close"
% (as specified by lineskiplimit) to the line before it. The the prevdepth,
% lineskip limit, and lineskip optional parameters are not needed for the
% first object/line on a page (i.e., prevdepth <= -1000pt) where the simplier
% \topskip spacing rules are in effect.
%
% Offset is a manual adjustment that is added to the height calculations of
% object irrespective of the value of \prevdepth. It is useful when the top
% baselineskip will result in a noninteger unit height object placement even
% if the object itself has integral height. e.g., a footnotesize baselineskip
% is used before the object, thus an offset of, say -3pt, can be given as a
% correction.

% Common combinations of these parameters include:
%
% top baselineskip:   (and default values for offset, prevdepth, etc.)
% \topskip            % for objects that appear at the top of a page
% \maxdimen           % always ignore the height of the top line
% 0pt                 % always consider any positive height of the top line
%
% for objects to appear inline in normal text:
% top baselineskip = \baselineskip
%
% set prevdepth = -1000pt and top baselineskip = 0pt to consider the
% overall height of the object without any other external skip
% consideration

\newif\if@IEEEquantizevspacestarform       % flag to indicate star form
\newif\if@IEEEquantizevspaceuseoutdimenreg % flag to indicate output dimen register is to be used
% Use our own private registers because the object could contain a
% structure that uses the existing tmp scratch pad registers
\newdimen\@IEEEquantizeheightA
\newdimen\@IEEEquantizeheightB
\newdimen\@IEEEquantizeheightC
\newdimen\@IEEEquantizeprevdepth % need to save this early as can change
\newcount\@IEEEquantizemultiple
\newbox\@IEEEquantizeboxA


\def\@IEEEquantizevspace{\@ifnextchar [{\@IEEEquantizevspaceuseoutdimenregtrue\@@IEEEquantizevspace}{\@IEEEquantizevspaceuseoutdimenregfalse\@@IEEEquantizevspace[]}}


\long\def\@@IEEEquantizevspace[#1]#2{\relax
% acquire and store
% #1 optional output dimen register
% #2 object
\edef\@IEEEquantizeoutdimenreg{#1}\relax
% allow for object specifications that contain parameters
\@IEEEtrantmptoksA={#2}\relax
\long\edef\@IEEEquantizeobject{\the\@IEEEtrantmptoksA}\relax
\@ifnextchar [{\@@@IEEEquantizevspace}{\@@@IEEEquantizevspace[\relax]}}

\long\def\@@@IEEEquantizevspace[#1]#2{\relax
% acquire and store
% [#1] optional object decl, is \relax if not given by user
% #2 top baselineskip
% allow for object decl specifications that have parameters
\@IEEEtrantmptoksA={#1}\relax
\long\edef\@IEEEquantizeobjectdecl{\the\@IEEEtrantmptoksA}\relax
\edef\@IEEEquantizetopbaselineskip{#2}\ivIEEEquantizevspace}

% acquire optional argument set and store
% [offset][prevdepth][lineskip limit][lineskip]
\def\ivIEEEquantizevspace{\@ifnextchar [{\@vIEEEquantizevspace}{\@vIEEEquantizevspace[0pt]}}
\def\@vIEEEquantizevspace[#1]{\edef\@IEEEquantizeoffset{#1}\@ifnextchar [{\@viIEEEquantizevspace}{\@viIEEEquantizevspace[\prevdepth]}}
\def\@viIEEEquantizevspace[#1]{\@IEEEquantizeprevdepth=#1\relax\@ifnextchar [{\@viiIEEEquantizevspace}{\@viiIEEEquantizevspace[\lineskiplimit]}}
\def\@viiIEEEquantizevspace[#1]{\edef\@IEEEquantizelineskiplimit{#1}\@ifnextchar [{\@viiiIEEEquantizevspace}{\@viiiIEEEquantizevspace[\lineskip]}}
\def\@viiiIEEEquantizevspace[#1]{\edef\@IEEEquantizelineskip{#1}\@ixIEEEquantizevspace}

% main routine
\def\@ixIEEEquantizevspace#1#2#3{\relax
\edef\@IEEEquantizeunitheight{#1}\relax
\edef\@IEEEquantizenomvspace{#2}\relax
\edef\@IEEEquantizeminvspace{#3}\relax
% \@IEEEquantizeoutdimenreg
% \@IEEEquantizeobject
% \@IEEEquantizeobjectdecl
% \@IEEEquantizetopbaselineskip
% \@IEEEquantizeoffset
% \@IEEEquantizeprevdepth
% \@IEEEquantizelineskiplimit
% \@IEEEquantizelineskip
% \@IEEEquantizeunitheight
% \@IEEEquantizenomvspace
% \@IEEEquantizeminvspace
% get overall height of object
\setbox\@IEEEquantizeboxA\vbox{\begingroup\@IEEEquantizeobjectdecl\@IEEEquantizeobject\relax\endgroup}\relax
\@IEEEquantizeheightA\ht\@IEEEquantizeboxA\relax
% get height of first line of object
\setbox\@IEEEquantizeboxA\vtop{\begingroup\@IEEEquantizeobjectdecl\@IEEEquantizeobject\relax\endgroup}\relax
\@IEEEquantizeheightB\ht\@IEEEquantizeboxA\relax
\ifdim\@IEEEquantizeprevdepth>-1000pt\relax % prevdepth > -1000pf means full baselineskip\lineskip rules in effect
% lineskip spacing rule takes effect if height of top line > baselineskip - prevdepth - lineskiplimit, 
% otherwise the baselineskip rule is in effect and the height of the first line does not matter at all.
\@IEEEquantizeheightC=\@IEEEquantizetopbaselineskip\relax
\advance\@IEEEquantizeheightC-\@IEEEquantizeprevdepth\relax
\advance\@IEEEquantizeheightC-\@IEEEquantizelineskiplimit\relax % this works even though \@IEEEquantizelineskiplimit is a macro because TeX allows --10pt notation
\ifdim\@IEEEquantizeheightB>\@IEEEquantizeheightC\relax
% lineskip spacing rule is in effect i.e., the object is going to be shifted down relative to the
% baselineskip set position by its top line height (already a part of the total height) + prevdepth + lineskip - baselineskip
\advance\@IEEEquantizeheightA\@IEEEquantizeprevdepth\relax
\advance\@IEEEquantizeheightA\@IEEEquantizelineskip\relax
\advance\@IEEEquantizeheightA-\@IEEEquantizetopbaselineskip\relax
\else
% height of first line <= \@IEEEquantizetopbaselineskip - \@IEEEquantizeprevdepth - \@IEEEquantizelineskiplimit
% standard baselineskip rules are in effect, so don't consider height of first line
\advance\@IEEEquantizeheightA-\@IEEEquantizeheightB\relax
\fi
%
\else % prevdepth <= -1000pt, simplier \topskip type rules in effect
\ifdim\@IEEEquantizeheightB>\@IEEEquantizetopbaselineskip
% height of top line (already included in the total height) in excess of
% baselineskip is the amount it will be downshifted
\advance\@IEEEquantizeheightA-\@IEEEquantizetopbaselineskip\relax
\else
% height of first line is irrelevant, remove it
\advance\@IEEEquantizeheightA-\@IEEEquantizeheightB\relax
\fi
\fi % prevdepth <= -1000pt
%
% adjust height for any manual offset
\advance\@IEEEquantizeheightA\@IEEEquantizeoffset\relax
% add in nominal spacer
\advance\@IEEEquantizeheightA\@IEEEquantizenomvspace\relax
% check for nonzero unitheight
\@IEEEquantizeheightB=\@IEEEquantizeunitheight\relax
\ifnum\@IEEEquantizeheightB=0\relax
\@IEEEclspkgerror{IEEEquantizevspace unit height cannot be zero. Assuming 10pt.}%
{Division by zero is not allowed.}
\@IEEEquantizeheightB=10pt\relax
\fi
% get integer number of lines
\@IEEEquantizemultiple=\@IEEEquantizeheightA\relax
\divide\@IEEEquantizemultiple\@IEEEquantizeheightB\relax
% set A to contain the excess height over the \@IEEEquantizemultiple of lines
% A = height - multiple*unitheight
\@IEEEquantizeheightC\@IEEEquantizeheightB\relax
\multiply\@IEEEquantizeheightC\@IEEEquantizemultiple\relax
\advance\@IEEEquantizeheightA-\@IEEEquantizeheightC\relax
% set B to contain the height short of \@IEEEquantizemultiple+1 of lines
% B = unitheight - A
\advance\@IEEEquantizeheightB-\@IEEEquantizeheightA\relax
% choose A or B based on which is closer
\@IEEEquantizeheightC\@IEEEquantizenomvspace\relax
\ifdim\@IEEEquantizeheightA<\@IEEEquantizeheightB\relax
% C = nomvspace - A, go with lower
\advance\@IEEEquantizeheightC-\@IEEEquantizeheightA\relax
\else
% C = nomvspace + B, go with upper
\advance\@IEEEquantizeheightC\@IEEEquantizeheightB\relax
\fi
% if violate lower bound, use next integer bound
\ifdim\@IEEEquantizeheightC<\@IEEEquantizeminvspace\relax
% A + B = unitheight
\advance\@IEEEquantizeheightC\@IEEEquantizeheightA\relax
\advance\@IEEEquantizeheightC\@IEEEquantizeheightB\relax
\fi
% export object and spacer outside of group
\global\let\@IEEEquantizeobjectout\@IEEEquantizeobject\relax
\global\@IEEEquantizeheightC\@IEEEquantizeheightC\relax
\endgroup
\if@IEEEquantizevspaceuseoutdimenreg
\@IEEEquantizeoutdimenreg=\@IEEEquantizeheightC\relax
\else
\@IEEEquantizeobjectout\relax
\vskip\@IEEEquantizeheightC\relax
\fi}


% user command to disable all global assignments, possible use within object decl
\def\IEEEquantizedisableglobal{\let\global\relax
\let\gdef\def
\let\xdef\edef}
% user command to allow for the disabling of \thanks and other commands, possible use within object decl
\def\IEEEquantizedisabletitlecmds{\long\def\thanks##1{\relax}\relax
\long\def\IEEEcompsocitemizethanks##1{\relax}\def\newpage{\relax}}





% V1.6
% we allow the user access to the last part of the title area
% useful in emergencies such as when a different spacing is needed
% This text is NOT compensated for in the dynamic sizer.
\let\@IEEEaftertitletext=\relax
\long\def\IEEEaftertitletext#1{\def\@IEEEaftertitletext{#1}}


% V1.7 provide a way for users to enter abstract and keywords
% into the onecolumn title are. This text is compensated for
% in the dynamic sizer.
\let\@IEEEtitleabstractindextext=\relax
\long\def\IEEEtitleabstractindextext#1{\def\@IEEEtitleabstractindextext{#1}}

% V1.7 provide a way for users to get the \@IEEEtitleabstractindextext if
% not in compsoc or transmag journal mode - this way abstract and keywords
% can still be placed in their conventional position if not in those modes.
\def\IEEEdisplaynontitleabstractindextext{%
% display for all conference formats
\ifCLASSOPTIONconference\@IEEEtitleabstractindextext\relax
\else% non-conferences
  % V1.8a display for all technotes
  \ifCLASSOPTIONtechnote\@IEEEtitleabstractindextext\relax
  % V1.8a add diamond line after abstract and index terms for compsoc technotes
  \@IEEEcompsoconly{\noindent\hfill\IEEEcompsocdiamondline\hfill\hbox{}\par}\relax
  \else % non-conferences and non-technotes
    \ifCLASSOPTIONcompsoc% display if not compsoc and not transmag
    \else
      \ifCLASSOPTIONtransmag
      \else% not compsoc journal nor transmag journal
         \@IEEEtitleabstractindextext\relax
      \fi
    \fi
  \fi
\fi}


% command to allow alteration of baselinestretch, but only if the current
% baselineskip is unity. Used to tweak the compsoc abstract and keywords line spacing.
\def\@IEEEtweakunitybaselinestretch#1{{\def\baselinestretch{1}\selectfont
\global\@tempskipa\baselineskip}\ifnum\@tempskipa=\baselineskip%
\def\baselinestretch{#1}\selectfont\fi\relax}


% abstract and keywords are in \small, except 
% for 9pt docs in which they are in \footnotesize
% Because 9pt docs use an 8pt footnotesize, \small
% becomes a rather awkward 8.5pt
\def\@IEEEabskeysecsize{\small}
\ifx\CLASSOPTIONpt\@IEEEptsizenine
 \def\@IEEEabskeysecsize{\footnotesize}
\fi

% compsoc journals use \footnotesize, compsoc conferences use normalsize
\@IEEEcompsoconly{\def\@IEEEabskeysecsize{\footnotesize}}
\@IEEEcompsocconfonly{\def\@IEEEabskeysecsize{\small}}


% V1.6 have abstract and keywords strip leading spaces, pars and newlines
% so that spacing is more tightly controlled.
\def\abstract{\normalfont
    \if@twocolumn
      \@IEEEabskeysecsize\bfseries\textit{\abstractname}---\relax
    \else
      \bgroup\par\addvspace{0.5\baselineskip}\centering\vspace{-1.78ex}\@IEEEabskeysecsize\textbf{\abstractname}\par\addvspace{0.5\baselineskip}\egroup\quotation\@IEEEabskeysecsize
    \fi\@IEEEgobbleleadPARNLSP}
% V1.6 The IEEE wants only 1 pica from end of abstract to introduction heading when in 
% conference mode (the heading already has this much above it)
\def\endabstract{\relax\ifCLASSOPTIONconference\vspace{0ex}\else\vspace{1.34ex}\fi\par\if@twocolumn\else\endquotation\fi
    \normalfont\normalsize}

\def\IEEEkeywords{\normalfont
    \if@twocolumn
      \@IEEEabskeysecsize\bfseries\textit{\IEEEkeywordsname}---\relax
    \else
      \bgroup\par\addvspace{0.5\baselineskip}\centering\@IEEEabskeysecsize\textbf{\IEEEkeywordsname}\par\addvspace{0.5\baselineskip}\egroup\quotation\@IEEEabskeysecsize
    \fi\@IEEEgobbleleadPARNLSP}
\def\endIEEEkeywords{\relax\ifCLASSOPTIONtechnote\vspace{1.34ex}\else\vspace{0.67ex}\fi
    \par\if@twocolumn\else\endquotation\fi%
    \normalfont\normalsize}

% V1.7 compsoc keywords index terms
\ifCLASSOPTIONcompsoc
  \ifCLASSOPTIONconference% compsoc conference
\def\abstract{\normalfont\@IEEEtweakunitybaselinestretch{1.15}\bfseries
    \if@twocolumn
      \@IEEEabskeysecsize\noindent\textit{\abstractname}---\relax
    \else
      \bgroup\par\addvspace{0.5\baselineskip}\centering\vspace{-1.78ex}\@IEEEabskeysecsize\textbf{\abstractname}\par\addvspace{0.5\baselineskip}\egroup\quotation\@IEEEabskeysecsize%
    \fi\@IEEEgobbleleadPARNLSP}
\def\IEEEkeywords{\normalfont\@IEEEtweakunitybaselinestretch{1.15}\bfseries
    \if@twocolumn
      \@IEEEabskeysecsize\vskip 0.5\baselineskip plus 0.25\baselineskip minus 0.25\baselineskip\noindent
      \textit{\IEEEkeywordsname}---\relax
    \else
      \bgroup\par\addvspace{0.5\baselineskip}\centering\@IEEEabskeysecsize\textbf{\IEEEkeywordsname}\par\addvspace{0.5\baselineskip}\egroup\quotation\@IEEEabskeysecsize%
    \fi\@IEEEgobbleleadPARNLSP}
  \else% compsoc not conference
\def\abstract{\normalfont\@IEEEtweakunitybaselinestretch{1.15}\sffamily
    \if@twocolumn
      \@IEEEabskeysecsize\noindent\textbf{\abstractname}---\relax
    \else
      \bgroup\par\addvspace{0.5\baselineskip}\centering\vspace{-1.78ex}\@IEEEabskeysecsize\textbf{\abstractname}\par\addvspace{0.5\baselineskip}\egroup\quotation\@IEEEabskeysecsize%
    \fi\@IEEEgobbleleadPARNLSP}
\def\IEEEkeywords{\normalfont\@IEEEtweakunitybaselinestretch{1.15}\sffamily
    \if@twocolumn
      \@IEEEabskeysecsize\vskip 0.5\baselineskip plus 0.25\baselineskip minus 0.25\baselineskip\noindent
      \textbf{\IEEEkeywordsname}---\relax
    \else
      \bgroup\par\addvspace{0.5\baselineskip}\centering\@IEEEabskeysecsize\textbf{\IEEEkeywordsname}\par\addvspace{0.5\baselineskip}\egroup\quotation\@IEEEabskeysecsize%
    \fi\@IEEEgobbleleadPARNLSP}
  \fi
\fi

% V1.8 transmag keywords index terms
% no abstract name, use indentation
\ifCLASSOPTIONtransmag
\def\abstract{\normalfont\parindent 1em\relax
    \if@twocolumn
      \@IEEEabskeysecsize\bfseries\indent
    \else
      \bgroup\par\addvspace{0.5\baselineskip}\centering\vspace{-1.78ex}\@IEEEabskeysecsize
      \textbf{\abstractname}\par\addvspace{0.5\baselineskip}\egroup\quotation\@IEEEabskeysecsize
    \fi\@IEEEgobbleleadPARNLSP}

\def\IEEEkeywords{\normalfont\parindent 1em\relax
    \if@twocolumn
      \@IEEEabskeysecsize\vspace{1\baselineskip}\bfseries\indent\textit{\IEEEkeywordsname}---\relax
    \else
      \bgroup\par\vspace{1\baselineskip}\centering\@IEEEabskeysecsize
      \textbf{\IEEEkeywordsname}\par\addvspace{0.5\baselineskip}\egroup\quotation\@IEEEabskeysecsize
    \fi\@IEEEgobbleleadPARNLSP}
\fi



% gobbles all leading \, \\ and \par, upon finding first token that
% is not a \ , \\ or a \par, it ceases and returns that token
% 
% used to strip leading \, \\ and \par from the input
% so that such things in the beginning of an environment will not
% affect the formatting of the text
\long\def\@IEEEgobbleleadPARNLSP#1{\let\@IEEEswallowthistoken=0%
\let\@IEEEgobbleleadPARNLSPtoken#1%
\let\@IEEEgobbleleadPARtoken=\par%
\let\@IEEEgobbleleadNLtoken=\\%
\let\@IEEEgobbleleadSPtoken=\ %
\def\@IEEEgobbleleadSPMACRO{\ }%
\ifx\@IEEEgobbleleadPARNLSPtoken\@IEEEgobbleleadPARtoken%
\let\@IEEEswallowthistoken=1%
\fi%
\ifx\@IEEEgobbleleadPARNLSPtoken\@IEEEgobbleleadNLtoken%
\let\@IEEEswallowthistoken=1%
\fi%
\ifx\@IEEEgobbleleadPARNLSPtoken\@IEEEgobbleleadSPtoken%
\let\@IEEEswallowthistoken=1%
\fi%
% a control space will come in as a macro
% when it is the last one on a line
\ifx\@IEEEgobbleleadPARNLSPtoken\@IEEEgobbleleadSPMACRO%
\let\@IEEEswallowthistoken=1%
\fi%
% if we have to swallow this token, do so and taste the next one
% else spit it out and stop gobbling
\ifx\@IEEEswallowthistoken 1\let\@IEEEnextgobbleleadPARNLSP=\@IEEEgobbleleadPARNLSP\else%
\let\@IEEEnextgobbleleadPARNLSP=#1\fi%
\@IEEEnextgobbleleadPARNLSP}%




% TITLING OF SECTIONS
\def\@IEEEsectpunct{:\ \,}  % Punctuation after run-in section heading  (headings which are
                            % part of the paragraphs), need little bit more than a single space
                            % spacing from section number to title
% compsoc conferences use regular period/space punctuation
\ifCLASSOPTIONcompsoc
\ifCLASSOPTIONconference
\def\@IEEEsectpunct{.\ }
\fi\fi


\def\@seccntformat#1{\csname the#1dis\endcsname\hskip 0.5em\relax}

\ifCLASSOPTIONcompsoc
% compsoc journals need extra spacing
\ifCLASSOPTIONconference\else
\def\@seccntformat#1{\csname the#1dis\endcsname\hskip 1em\relax}
\fi\fi

%v1.7 put {} after #6 to allow for some types of user font control
%and use \@@par rather than \par
\def\@sect#1#2#3#4#5#6[#7]#8{%
  \ifnum #2>\c@secnumdepth
     \let\@svsec\@empty
  \else
     \refstepcounter{#1}%
     % load section label and spacer into \@svsec
     \protected@edef\@svsec{\@seccntformat{#1}\relax}%
  \fi%
  \@tempskipa #5\relax
  \ifdim \@tempskipa>\z@% tempskipa determines whether is treated as a high
     \begingroup #6{\relax% or low level heading
      \noindent % subsections are NOT indented
       % print top level headings. \@svsec is label, #8 is heading title
       % The IEEE does not block indent the section title text, it flows like normal
       {\hskip #3\relax\@svsec}{\interlinepenalty \@M #8\@@par}}%
     \endgroup
     \addcontentsline{toc}{#1}{\ifnum #2>\c@secnumdepth\relax\else
               \protect\numberline{\csname the#1\endcsname}\fi#7}%
  \else % printout low level headings
     % svsechd seems to swallow the trailing space, protect it with \mbox{}
     % got rid of sectionmark stuff
     \def\@svsechd{#6{\hskip #3\relax\@svsec #8\@IEEEsectpunct\mbox{}}%
     \addcontentsline{toc}{#1}{\ifnum #2>\c@secnumdepth\relax\else
               \protect\numberline{\csname the#1\endcsname}\fi#7}}%
  \fi%skip down
  \@xsect{#5}}


% section* handler
%v1.7 put {} after #4 to allow for some types of user font control
%and use \@@par rather than \par
\def\@ssect#1#2#3#4#5{\@tempskipa #3\relax
  \ifdim \@tempskipa>\z@
     %\begingroup #4\@hangfrom{\hskip #1}{\interlinepenalty \@M #5\par}\endgroup
     % The IEEE does not block indent the section title text, it flows like normal
     \begingroup \noindent #4{\relax{\hskip #1}{\interlinepenalty \@M #5\@@par}}\endgroup
  % svsechd swallows the trailing space, protect it with \mbox{}
  \else \def\@svsechd{#4{\hskip #1\relax #5\@IEEEsectpunct\mbox{}}}\fi
  \@xsect{#3}}


%% SECTION heading spacing and font
%%
% arguments are: #1 - sectiontype name
% (for \@sect)   #2 - section level
%                #3 - section heading indent
%                #4 - top separation (absolute value used, neg indicates not to indent main text)
%                     If negative, make stretch parts negative too!
%                #5 - (absolute value used) positive: bottom separation after heading,
%                      negative: amount to indent main text after heading
%                Both #4 and #5 negative means to indent main text and use negative top separation
%                #6 - font control
% You've got to have \normalfont\normalsize in the font specs below to prevent
% trouble when you do something like:
% \section{Note}{\ttfamily TT-TEXT} is known to ... 
% The IEEE sometimes REALLY stretches the area before a section
% heading by up to about 0.5in. However, it may not be a good
% idea to let LaTeX have quite this much rubber.
\ifCLASSOPTIONconference%
% The IEEE wants section heading spacing to decrease for conference mode
\def\section{\@startsection{section}{1}{\z@}{1.5ex plus 1.5ex minus 0.5ex}%
{0.7ex plus 1ex minus 0ex}{\normalfont\normalsize\centering\scshape}}%
\def\subsection{\@startsection{subsection}{2}{\z@}{1.5ex plus 1.5ex minus 0.5ex}%
{0.7ex plus .5ex minus 0ex}{\normalfont\normalsize\itshape}}%
\else % for journals
\def\section{\@startsection{section}{1}{\z@}{3.0ex plus 1.5ex minus 1.5ex}% V1.6 3.0ex from 3.5ex
{0.7ex plus 1ex minus 0ex}{\normalfont\normalsize\centering\scshape}}%
\def\subsection{\@startsection{subsection}{2}{\z@}{3.5ex plus 1.5ex minus 1.5ex}%
{0.7ex plus .5ex minus 0ex}{\normalfont\normalsize\itshape}}%
\fi

% for both journals and conferences
% decided to put in a little rubber above the section, might help somebody
\def\subsubsection{\@startsection{subsubsection}{3}{\parindent}{0ex plus 0.1ex minus 0.1ex}%
{0ex}{\normalfont\normalsize\itshape}}%
\def\paragraph{\@startsection{paragraph}{4}{2\parindent}{0ex plus 0.1ex minus 0.1ex}%
{0ex}{\normalfont\normalsize\itshape}}%


% compsoc
\ifCLASSOPTIONcompsoc
\ifCLASSOPTIONconference
% compsoc conference
\def\section{\@startsection{section}{1}{\z@}{1\baselineskip plus 0.25\baselineskip minus 0.25\baselineskip}%
{1\baselineskip plus 0.25\baselineskip minus 0.25\baselineskip}{\normalfont\large\bfseries}}%
\def\subsection{\@startsection{subsection}{2}{\z@}{1\baselineskip plus 0.25\baselineskip minus 0.25\baselineskip}%
{1\baselineskip plus 0.25\baselineskip minus 0.25\baselineskip}{\normalfont\sublargesize\bfseries}}%
\def\subsubsection{\@startsection{subsubsection}{3}{\z@}{1\baselineskip plus 0.25\baselineskip minus 0.25\baselineskip}%
{0ex}{\normalfont\normalsize\bfseries}}%
\def\paragraph{\@startsection{paragraph}{4}{2\parindent}{0ex plus 0.1ex minus 0.1ex}%
{0ex}{\normalfont\normalsize}}%
\else% compsoc journals
% use negative top separation as compsoc journals do not indent paragraphs after section titles
\def\section{\@startsection{section}{1}{\z@}{-3.5ex plus -2ex minus -1.5ex}%
{0.7ex plus 1ex minus 0ex}{\normalfont\sublargesize\sffamily\bfseries\scshape}}%
% Note that subsection and smaller may not be correct for the Computer Society,
% I have to look up an example.
\def\subsection{\@startsection{subsection}{2}{\z@}{-3.5ex plus -1.5ex minus -1.5ex}%
{0.7ex plus .5ex minus 0ex}{\normalfont\normalsize\sffamily\bfseries}}%
\def\subsubsection{\@startsection{subsubsection}{3}{\z@}{-2.5ex plus -1ex minus -1ex}%
{0.5ex plus 0.5ex minus 0ex}{\normalfont\normalsize\sffamily\itshape}}%
\def\paragraph{\@startsection{paragraph}{4}{2\parindent}{-0ex plus -0.1ex minus -0.1ex}%
{0ex}{\normalfont\normalsize}}%
\fi\fi

% transmag
\ifCLASSOPTIONtransmag
\def\subsection{\@startsection{subsection}{2}{0.75\parindent}{3.5ex plus 1.5ex minus 1.5ex}%
{0.7ex plus .5ex minus 0ex}{\normalfont\normalsize\itshape}}%
\def\subsubsection{\@startsection{subsubsection}{3}{1.25\parindent}{0.1ex plus 0.1ex minus 0.1ex}%
{0.1ex}{\normalfont\normalsize\itshape}}%
\fi


% V1.8a provide for a raised line Introduction section for use with Computer
% Society papers. We have to remove any spacing glue after the section
% heading and then remove the blank line for the new paragraph after it.
% LaTeX's section handler alters \everypar and we need to propogate those
% changes outside of the \parbox lest there be spacing problems at the top
% of the next section.
\def\IEEEraisesectionheading#1{\noindent\raisebox{1.5\baselineskip}[0pt][0pt]{\parbox[b]{\columnwidth}{#1\unskip\global\everypar=\everypar}}\vspace{-1\baselineskip}\vspace{-\parskip}\par}



%% ENVIRONMENTS
% "box" symbols at end of proofs
\def\IEEEQEDclosed{\mbox{\rule[0pt]{1.3ex}{1.3ex}}} % for a filled box
% V1.6 some journals use an open box instead that will just fit around a closed one
\def\IEEEQEDopen{{\setlength{\fboxsep}{0pt}\setlength{\fboxrule}{0.2pt}\fbox{\rule[0pt]{0pt}{1.3ex}\rule[0pt]{1.3ex}{0pt}}}}
\ifCLASSOPTIONcompsoc
\def\IEEEQED{\IEEEQEDopen}   % default to open for compsoc
\else
\def\IEEEQED{\IEEEQEDclosed} % otherwise default to closed
\fi

%V1.8 flag to indicate if QED symbol is to be shown
\newif\if@IEEEQEDshow  \@IEEEQEDshowtrue
\def\IEEEproofindentspace{2\parindent}% V1.8 allow user to change indentation amount if desired
% v1.7 name change to avoid namespace collision with amsthm. Also add support
% for an optional argument.
\def\IEEEproof{\@ifnextchar[{\@IEEEproof}{\@IEEEproof[\IEEEproofname]}}
\def\@IEEEproof[#1]{\@IEEEQEDshowtrue\par\noindent\hspace{\IEEEproofindentspace}{\itshape #1: }}
\def\endIEEEproof{\if@IEEEQEDshow\hspace*{\fill}\nobreakspace\IEEEQED\fi\par}
% qedhere for equation environments, similar to AMS \qedhere
\def\IEEEQEDhereeqn{\global\@IEEEQEDshowfalse\eqno\let\eqno\relax\let\leqno\relax
                    \let\veqno\relax\hbox{\IEEEQED}}
% IEEE style qedhere for IEEEeqnarray and other environments
\def\IEEEQEDhere{\global\@IEEEQEDshowfalse\IEEEQED}
% command to disable QED at end of IEEEproof
\def\IEEEQEDoff{\global\@IEEEQEDshowfalse}


%\itemindent is set to \z@ by list, so define new temporary variable
\newdimen\@IEEEtmpitemindent

\ifCLASSOPTIONcompsoc
% V1.8a compsoc uses bold theorem titles, a period instead of a colon, vertical spacing, and hanging indentation
% V1.8 allow long theorem names to break across lines.
% Thanks to Miquel Payaro for reporting this.
\def\@begintheorem#1#2{\@IEEEtmpitemindent\itemindent\relax
    \topsep 0.2\@IEEEnormalsizeunitybaselineskip plus 0.26\@IEEEnormalsizeunitybaselineskip minus 0.05\@IEEEnormalsizeunitybaselineskip
    \rmfamily\trivlist\hangindent\parindent%
    \item[]\textit{\bfseries\noindent #1\ #2.} \itemindent\@IEEEtmpitemindent\relax}
\def\@opargbegintheorem#1#2#3{\@IEEEtmpitemindent\itemindent\relax
\topsep 0.2\@IEEEnormalsizeunitybaselineskip plus 0.26\@IEEEnormalsizeunitybaselineskip minus 0.05\@IEEEnormalsizeunitybaselineskip
\rmfamily\trivlist\hangindent\parindent%
% V1.6 The IEEE is back to using () around theorem names which are also in italics
% Thanks to Christian Peel for reporting this.
    \item[]\textit{\bfseries\noindent #1\ #2\ (#3).} \itemindent\@IEEEtmpitemindent\relax}
% V1.7 remove bogus \unskip that caused equations in theorems to collide with
% lines below.
\def\@endtheorem{\endtrivlist\vskip 0.25\@IEEEnormalsizeunitybaselineskip plus 0.26\@IEEEnormalsizeunitybaselineskip minus 0.05\@IEEEnormalsizeunitybaselineskip}
\else
%
% noncompsoc
% 
% V1.8 allow long theorem names to break across lines.
% Thanks to Miquel Payaro for reporting this.
\def\@begintheorem#1#2{\@IEEEtmpitemindent\itemindent\relax\topsep 0pt\rmfamily\trivlist%
    \item[]\textit{\indent #1\ #2:} \itemindent\@IEEEtmpitemindent\relax}
\def\@opargbegintheorem#1#2#3{\@IEEEtmpitemindent\itemindent\relax\topsep 0pt\rmfamily \trivlist%
% V1.6 The IEEE is back to using () around theorem names which are also in italics
% Thanks to Christian Peel for reporting this.
    \item[]\textit{\indent #1\ #2\ (#3):} \itemindent\@IEEEtmpitemindent\relax}
% V1.7 remove bogus \unskip that caused equations in theorems to collide with
% lines below.
\def\@endtheorem{\endtrivlist}
\fi



% V1.6
% display command for the section the theorem is in - so that \thesection
% is not used as this will be in Roman numerals when we want arabic.
% LaTeX2e uses \def\@thmcounter#1{\noexpand\arabic{#1}} for the theorem number
% (second part) display and \def\@thmcountersep{.} as a separator.
% V1.7 intercept calls to the section counter and reroute to \@IEEEthmcounterinsection
% to allow \appendix(ices} to override as needed.
%
% special handler for sections, allows appendix(ices) to override
\gdef\@IEEEthmcounterinsection#1{\arabic{#1}}
% string macro
\edef\@IEEEstringsection{section}

% redefine the #1#2[#3] form of newtheorem to use a hook to \@IEEEthmcounterinsection
% if section in_counter is used
\def\@xnthm#1#2[#3]{%
  \expandafter\@ifdefinable\csname #1\endcsname
    {\@definecounter{#1}\@newctr{#1}[#3]%
     \edef\@IEEEstringtmp{#3}
     \ifx\@IEEEstringtmp\@IEEEstringsection
     \expandafter\xdef\csname the#1\endcsname{%
     \noexpand\@IEEEthmcounterinsection{#3}\@thmcountersep
          \@thmcounter{#1}}%
     \else
     \expandafter\xdef\csname the#1\endcsname{%
       \expandafter\noexpand\csname the#3\endcsname \@thmcountersep
          \@thmcounter{#1}}%
     \fi
     \global\@namedef{#1}{\@thm{#1}{#2}}%
     \global\@namedef{end#1}{\@endtheorem}}}



%% SET UP THE DEFAULT PAGESTYLE
\pagestyle{headings}
\pagenumbering{arabic}

% normally the page counter starts at 1
\setcounter{page}{1}
% however, for peerreview the cover sheet is page 0 or page -1
% (for duplex printing)
\ifCLASSOPTIONpeerreview
   \if@twoside
      \setcounter{page}{-1}
   \else
      \setcounter{page}{0}
   \fi
\fi

% standard book class behavior - let bottom line float up and down as
% needed when single sided
\ifCLASSOPTIONtwoside\else\raggedbottom\fi
% if two column - turn on twocolumn, allow word spacings to stretch more and
% enforce a rigid position for the last lines
\ifCLASSOPTIONtwocolumn
% the peer review option delays invoking twocolumn
   \ifCLASSOPTIONpeerreview\else
      \twocolumn
   \fi
\sloppy 
\flushbottom
\fi




% \APPENDIX and \APPENDICES definitions

% This is the \@ifmtarg command from the LaTeX ifmtarg package
% by Peter Wilson (CUA) and Donald Arseneau
% \@ifmtarg is used to determine if an argument to a command
% is present or not.
% For instance:
% \@ifmtarg{#1}{\typeout{empty}}{\typeout{has something}}
% \@ifmtarg is used with our redefined \section command if
% \appendices is invoked.
% The command \section will behave slightly differently depending
% on whether the user specifies a title: 
% \section{My appendix title}
% or not:
% \section{}
% This way, we can eliminate the blank lines where the title
% would be, and the unneeded : after Appendix in the table of
% contents 
\begingroup
\catcode`\Q=3
\long\gdef\@ifmtarg#1{\@xifmtarg#1QQ\@secondoftwo\@firstoftwo\@nil}
\long\gdef\@xifmtarg#1#2Q#3#4#5\@nil{#4}
\endgroup
% end of \@ifmtarg defs


% V1.7
% command that allows the one time saving of the original definition
% of section to \@IEEEappendixsavesection for \appendix or \appendices 
% we don't save \section here as it may be redefined later by other
% packages (hyperref.sty, etc.)
\def\@IEEEsaveoriginalsectiononce{\let\@IEEEappendixsavesection\section
\let\@IEEEsaveoriginalsectiononce\relax}

% neat trick to grab and process the argument from \section{argument}
% we process differently if the user invoked \section{} with no
% argument (title)
% note we reroute the call to the old \section*
\def\@IEEEprocessthesectionargument#1{%
\@ifmtarg{#1}{%
\@IEEEappendixsavesection*{\appendixname\nobreakspace\thesectiondis}%
\addcontentsline{toc}{section}{\appendixname\nobreakspace\thesection}}{%
\@IEEEappendixsavesection*{\appendixname\nobreakspace\thesectiondis\\* #1}%
\addcontentsline{toc}{section}{\appendixname\nobreakspace\thesection: #1}}}

% we use this if the user calls \section{} after
% \appendix-- which has no meaning. So, we ignore the
% command and its argument. Then, warn the user.
\def\@IEEEdestroythesectionargument#1{\typeout{** WARNING: Ignoring useless
\protect\section\space in Appendix (line \the\inputlineno).}}


% remember \thesection forms will be displayed in \ref calls
% and in the Table of Contents.
% The \sectiondis form is used in the actual heading itself

% appendix command for one single appendix
% normally has no heading. However, if you want a 
% heading, you can do so via the optional argument:
% \appendix[Optional Heading]
\def\appendix{\relax}
\renewcommand{\appendix}[1][]{\@IEEEsaveoriginalsectiononce\par
    % v1.6 keep hyperref's identifiers unique
    \gdef\theHsection{Appendix.A}%
    % v1.6 adjust hyperref's string name for the section
    \xdef\Hy@chapapp{appendix}%
    \setcounter{section}{0}%
    \setcounter{subsection}{0}%
    \setcounter{subsubsection}{0}%
    \setcounter{paragraph}{0}%
    \gdef\thesection{A}%
    \gdef\thesectiondis{}% 
    \gdef\thesubsection{\Alph{subsection}}%
    \gdef\@IEEEthmcounterinsection##1{A}
    \refstepcounter{section}% update the \ref counter
    \@ifmtarg{#1}{\@IEEEappendixsavesection*{\appendixname}%
                  \addcontentsline{toc}{section}{\appendixname}}{%
             \@IEEEappendixsavesection*{\appendixname\nobreakspace\\* #1}%
             \addcontentsline{toc}{section}{\appendixname: #1}}%
    % redefine \section command for appendix
    % leave \section* as is
    \def\section{\@ifstar{\@IEEEappendixsavesection*}{%
                    \@IEEEdestroythesectionargument}}% throw out the argument
                                                     % of the normal form
}



% appendices command for multiple appendices
% user then calls \section with an argument (possibly empty) to
% declare the individual appendices
\def\appendices{\@IEEEsaveoriginalsectiononce\par
    % v1.6 keep hyperref's identifiers unique
    \gdef\theHsection{Appendix.\Alph{section}}%
    % v1.6 adjust hyperref's string name for the section
    \xdef\Hy@chapapp{appendix}%
    \setcounter{section}{-1}% we want \refstepcounter to use section 0
    \setcounter{subsection}{0}%
    \setcounter{subsubsection}{0}%
    \setcounter{paragraph}{0}%
    \ifCLASSOPTIONromanappendices%
    \gdef\thesection{\Roman{section}}%
    \gdef\thesectiondis{\Roman{section}}%
    \@IEEEcompsocconfonly{\gdef\thesectiondis{\Roman{section}.}}%
    \gdef\@IEEEthmcounterinsection##1{A\arabic{##1}}
    \else%
    \gdef\thesection{\Alph{section}}%
    \gdef\thesectiondis{\Alph{section}}%
    \@IEEEcompsocconfonly{\gdef\thesectiondis{\Alph{section}.}}%
    \gdef\@IEEEthmcounterinsection##1{\Alph{##1}}
    \fi%
    \refstepcounter{section}% update the \ref counter
    \setcounter{section}{0}% NEXT \section will be the FIRST appendix
    % redefine \section command for appendices
    % leave \section* as is
    \def\section{\@ifstar{\@IEEEappendixsavesection*}{% process the *-form
                    \refstepcounter{section}% or is a new section so,
                    \@IEEEprocessthesectionargument}}% process the argument 
                                                 % of the normal form
}



% V1.7 compoc uses nonbold drop cap and small caps word style
\ifCLASSOPTIONcompsoc
  \def\IEEEPARstartFONTSTYLE{\mdseries}
  \def\IEEEPARstartWORDFONTSTYLE{\scshape}
  \def\IEEEPARstartWORDCAPSTYLE{\relax}
\fi
%
%
% \IEEEPARstart
% Definition for the big two line drop cap letter at the beginning of the
% first paragraph of journal papers. The first argument is the first letter
% of the first word, the second argument is the remaining letters of the
% first word which will be rendered in upper case.
% In V1.6 this has been completely rewritten to:
% 
% 1. no longer have problems when the user begins an environment
%    within the paragraph that uses \IEEEPARstart.
% 2. auto-detect and use the current font family
% 3. revise handling of the space at the end of the first word so that
%    interword glue will now work as normal.
% 4. produce correctly aligned edges for the (two) indented lines.
% 
% We generalize things via control macros - playing with these is fun too.
% 
% V1.7 added more control macros to make it easy for IEEEtrantools.sty users
% to change the font style.
% 
% the number of lines that are indented to clear it
% may need to increase if using decenders
\providecommand{\IEEEPARstartDROPLINES}{2}
% minimum number of lines left on a page to allow a \@IEEEPARstart
% Does not take into consideration rubber shrink, so it tends to
% be overly cautious
\providecommand{\IEEEPARstartMINPAGELINES}{2}
% V1.7 the height of the drop cap is adjusted to match the height of this text
% in the current font (when \IEEEPARstart is called).
\providecommand{\IEEEPARstartHEIGHTTEXT}{T}
% the depth the letter is lowered below the baseline
% the height (and size) of the letter is determined by the sum
% of this value and the height of the \IEEEPARstartHEIGHTTEXT in the current
% font. It is a good idea to set this value in terms of the baselineskip
% so that it can respond to changes therein.
\providecommand{\IEEEPARstartDROPDEPTH}{1.1\baselineskip}
% V1.7 the font the drop cap will be rendered in,
% can take zero or one argument.
\providecommand{\IEEEPARstartFONTSTYLE}{\bfseries}
% V1.7 any additional, non-font related commands needed to modify
% the drop cap letter, can take zero or one argument.
\providecommand{\IEEEPARstartCAPSTYLE}{\MakeUppercase}
% V1.7 the font that will be used to render the rest of the word,
% can take zero or one argument.
\providecommand{\IEEEPARstartWORDFONTSTYLE}{\relax}
% V1.7 any additional, non-font related commands needed to modify
% the rest of the word, can take zero or one argument.
\providecommand{\IEEEPARstartWORDCAPSTYLE}{\MakeUppercase}
% This is the horizontal separation distance from the drop letter to the main text.
% Lengths that depend on the font (e.g., ex, em, etc.) will be referenced
% to the font that is active when \IEEEPARstart is called. 
\providecommand{\IEEEPARstartSEP}{0.15em}
% V1.7 horizontal offset applied to the left of the drop cap.
\providecommand{\IEEEPARstartHOFFSET}{0em}
% V1.7 Italic correction command applied at the end of the drop cap.
\providecommand{\IEEEPARstartITLCORRECT}{\/}

% width of the letter output, set globally. Can be used in \IEEEPARstartSEP
% or \IEEEPARstartHOFFSET, but not the height lengths.
\newdimen\IEEEPARstartletwidth
\IEEEPARstartletwidth 0pt\relax

% definition of \IEEEPARstart
% THIS IS A CONTROLLED SPACING AREA, DO NOT ALLOW SPACES WITHIN THESE LINES
% 
% The token \@IEEEPARstartfont will be globally defined after the first use
% of \IEEEPARstart and will be a font command which creates the big letter
% The first argument is the first letter of the first word and the second
% argument is the rest of the first word(s).
\def\IEEEPARstart#1#2{\par{%
% if this page does not have enough space, break it and lets start
% on a new one
\@IEEEtranneedspace{\IEEEPARstartMINPAGELINES\baselineskip}{\relax}%
% V1.7 move this up here in case user uses \textbf for \IEEEPARstartFONTSTYLE
% which uses command \leavevmode which causes an unwanted \indent to be issued
\noindent
% calculate the desired height of the big letter
% it extends from the top of \IEEEPARstartHEIGHTTEXT in the current font
% down to \IEEEPARstartDROPDEPTH below the current baseline
\settoheight{\@IEEEtrantmpdimenA}{\IEEEPARstartHEIGHTTEXT}%
\addtolength{\@IEEEtrantmpdimenA}{\IEEEPARstartDROPDEPTH}%
% extract the name of the current font in bold
% and place it in \@IEEEPARstartFONTNAME
\def\@IEEEPARstartGETFIRSTWORD##1 ##2\relax{##1}%
{\IEEEPARstartFONTSTYLE{\selectfont\edef\@IEEEPARstartFONTNAMESPACE{\fontname\font\space}%
\xdef\@IEEEPARstartFONTNAME{\expandafter\@IEEEPARstartGETFIRSTWORD\@IEEEPARstartFONTNAMESPACE\relax}}}%
% define a font based on this name with a point size equal to the desired
% height of the drop letter
\font\@IEEEPARstartsubfont\@IEEEPARstartFONTNAME\space at \@IEEEtrantmpdimenA\relax%
% save this value as a counter (integer) value (sp points)
\@IEEEtrantmpcountA=\@IEEEtrantmpdimenA%
% now get the height of the actual letter produced by this font size
\settoheight{\@IEEEtrantmpdimenB}{\@IEEEPARstartsubfont\IEEEPARstartCAPSTYLE{#1}}%
% If something bogus happens like the first argument is empty or the
% current font is strange, do not allow a zero height.
\ifdim\@IEEEtrantmpdimenB=0pt\relax%
\typeout{** WARNING: IEEEPARstart drop letter has zero height! (line \the\inputlineno)}%
\typeout{ Forcing the drop letter font size to 10pt.}%
\@IEEEtrantmpdimenB=10pt%
\fi%
% and store it as a counter
\@IEEEtrantmpcountB=\@IEEEtrantmpdimenB%
% Since a font size doesn't exactly correspond to the height of the capital
% letters in that font, the actual height of the letter, \@IEEEtrantmpcountB,
% will be less than that desired, \@IEEEtrantmpcountA
% we need to raise the font size, \@IEEEtrantmpdimenA 
% by \@IEEEtrantmpcountA / \@IEEEtrantmpcountB
% But, TeX doesn't have floating point division, so we have to use integer
% division. Hence the use of the counters.
% We need to reduce the denominator so that the loss of the remainder will
% have minimal affect on the accuracy of the result
\divide\@IEEEtrantmpcountB by 200%
\divide\@IEEEtrantmpcountA by \@IEEEtrantmpcountB%
% Then reequalize things when we use TeX's ability to multiply by
% floating point values
\@IEEEtrantmpdimenB=0.005\@IEEEtrantmpdimenA%
\multiply\@IEEEtrantmpdimenB by \@IEEEtrantmpcountA%
% \@IEEEPARstartfont is globaly set to the calculated font of the big letter
% We need to carry this out of the local calculation area to to create the
% big letter.
\global\font\@IEEEPARstartfont\@IEEEPARstartFONTNAME\space at \@IEEEtrantmpdimenB%
% Now set \@IEEEtrantmpdimenA to the width of the big letter
% We need to carry this out of the local calculation area to set the
% hanging indent
\settowidth{\global\@IEEEtrantmpdimenA}{\@IEEEPARstartfont
\IEEEPARstartCAPSTYLE{#1\IEEEPARstartITLCORRECT}}}%
% end of the isolated calculation environment
\global\IEEEPARstartletwidth\@IEEEtrantmpdimenA\relax%
% add in the extra clearance we want
\advance\@IEEEtrantmpdimenA by \IEEEPARstartSEP\relax%
% add in the optional offset
\advance\@IEEEtrantmpdimenA by \IEEEPARstartHOFFSET\relax%
% V1.7 don't allow negative offsets to produce negative hanging indents
\@IEEEtrantmpdimenB\@IEEEtrantmpdimenA
\ifnum\@IEEEtrantmpdimenB < 0 \@IEEEtrantmpdimenB 0pt\fi
% \@IEEEtrantmpdimenA has the width of the big letter plus the
% separation space and \@IEEEPARstartfont is the font we need to use
% Now, we make the letter and issue the hanging indent command
% The letter is placed in a box of zero width and height so that other
% text won't be displaced by it.
\hangindent\@IEEEtrantmpdimenB\hangafter=-\IEEEPARstartDROPLINES%
\makebox[0pt][l]{\hspace{-\@IEEEtrantmpdimenA}%
\raisebox{-\IEEEPARstartDROPDEPTH}[0pt][0pt]{\hspace{\IEEEPARstartHOFFSET}%
\@IEEEPARstartfont\IEEEPARstartCAPSTYLE{#1\IEEEPARstartITLCORRECT}%
\hspace{\IEEEPARstartSEP}}}%
{\IEEEPARstartWORDFONTSTYLE{\IEEEPARstartWORDCAPSTYLE{\selectfont#2}}}}




% determines if the space remaining on a given page is equal to or greater
% than the specified space of argument one
% if not, execute argument two (only if the remaining space is greater than zero)
% and issue a \newpage
% 
% example: \@IEEEtranneedspace{2in}{\vfill}
% 
% Does not take into consideration rubber shrinkage, so it tends to
% be overly cautious
% Based on an example posted by Donald Arseneau
% Note this macro uses \@IEEEtrantmpdimenB internally for calculations,
% so DO NOT PASS \@IEEEtrantmpdimenB to this routine
% if you need a dimen register, import with \@IEEEtrantmpdimenA instead
\def\@IEEEtranneedspace#1#2{\penalty-100\begingroup%shield temp variable
\@IEEEtrantmpdimenB\pagegoal\advance\@IEEEtrantmpdimenB-\pagetotal% space left
\ifdim #1>\@IEEEtrantmpdimenB\relax% not enough space left
\ifdim\@IEEEtrantmpdimenB>\z@\relax #2\fi%
\newpage%
\fi\endgroup}



% IEEEbiography ENVIRONMENT
% Allows user to enter biography leaving place for picture (adapts to font size)
% As of V1.5, a new optional argument allows you to have a real graphic!
% V1.5 and later also fixes the "colliding biographies" which could happen when a 
% biography's text was shorter than the space for the photo.
% MDS 7/2001
% V1.6 prevent multiple biographies from making multiple TOC entries
\newif\if@IEEEbiographyTOCentrynotmade
\global\@IEEEbiographyTOCentrynotmadetrue

% biography counter so hyperref can jump directly to the biographies
% and not just the previous section
\newcounter{IEEEbiography}
\setcounter{IEEEbiography}{0}

% photo area size
\def\@IEEEBIOphotowidth{1.0in}    % width of the biography photo area
\def\@IEEEBIOphotodepth{1.25in}   % depth (height) of the biography photo area
% area cleared for photo
\def\@IEEEBIOhangwidth{1.14in}    % width cleared for the biography photo area
\def\@IEEEBIOhangdepth{1.25in}    % depth cleared for the biography photo area
                                  % actual depth will be a multiple of 
                                  % \baselineskip, rounded up
\def\@IEEEBIOskipN{4\baselineskip}% nominal value of the vskip above the biography

\newenvironment{IEEEbiography}[2][]{\normalfont\@IEEEcompsoconly{\sffamily}\footnotesize%
\unitlength 1in\parskip=0pt\par\parindent 1em\interlinepenalty500%
% we need enough space to support the hanging indent
% the nominal value of the spacer
% and one extra line for good measure
\@IEEEtrantmpdimenA=\@IEEEBIOhangdepth%
\advance\@IEEEtrantmpdimenA by \@IEEEBIOskipN%
\advance\@IEEEtrantmpdimenA by 1\baselineskip%
% if this page does not have enough space, break it and lets start
% with a new one
\@IEEEtranneedspace{\@IEEEtrantmpdimenA}{\relax}%
% nominal spacer can strech, not shrink use 1fil so user can out stretch with \vfill
\vskip \@IEEEBIOskipN plus 1fil minus 0\baselineskip%
% the default box for where the photo goes
\def\@IEEEtempbiographybox{{\setlength{\fboxsep}{0pt}\framebox{%
\begin{minipage}[b][\@IEEEBIOphotodepth][c]{\@IEEEBIOphotowidth}\centering PLACE\\ PHOTO\\ HERE \end{minipage}}}}%
%
% detect if the optional argument was supplied, this requires the
% \@ifmtarg command as defined in the appendix section above
% and if so, override the default box with what they want
\@ifmtarg{#1}{\relax}{\def\@IEEEtempbiographybox{\mbox{\begin{minipage}[b][\@IEEEBIOphotodepth][c]{\@IEEEBIOphotowidth}%
\centering%
#1%
\end{minipage}}}}% end if optional argument supplied
% Make an entry into the table of contents only if we have not done so before
\if@IEEEbiographyTOCentrynotmade%
% link labels to the biography counter so hyperref will jump
% to the biography, not the previous section
\setcounter{IEEEbiography}{-1}%
\refstepcounter{IEEEbiography}%
\addcontentsline{toc}{section}{Biographies}%
\global\@IEEEbiographyTOCentrynotmadefalse%
\fi%
% one more biography
\refstepcounter{IEEEbiography}%
% Make an entry for this name into the table of contents 
\addcontentsline{toc}{subsection}{#2}%
% V1.6 properly handle if a new paragraph should occur while the
% hanging indent is still active. Do this by redefining \par so
% that it will not start a new paragraph. (But it will appear to the
% user as if it did.) Also, strip any leading pars, newlines, or spaces.
\let\@IEEEBIOORGparCMD=\par% save the original \par command
\edef\par{\hfil\break\indent}% the new \par will not be a "real" \par
\settoheight{\@IEEEtrantmpdimenA}{\@IEEEtempbiographybox}% get height of biography box
\@IEEEtrantmpdimenB=\@IEEEBIOhangdepth%
\@IEEEtrantmpcountA=\@IEEEtrantmpdimenB% countA has the hang depth
\divide\@IEEEtrantmpcountA by \baselineskip%  calculates lines needed to produce the hang depth
\advance\@IEEEtrantmpcountA by 1% ensure we overestimate
% set the hanging indent
\hangindent\@IEEEBIOhangwidth%
\hangafter-\@IEEEtrantmpcountA%
% reference the top of the photo area to the top of a capital T
\settoheight{\@IEEEtrantmpdimenB}{\mbox{T}}%
% set the photo box, give it zero width and height so as not to disturb anything
\noindent\makebox[0pt][l]{\hspace{-\@IEEEBIOhangwidth}\raisebox{\@IEEEtrantmpdimenB}[0pt][0pt]{%
\raisebox{-\@IEEEBIOphotodepth}[0pt][0pt]{\@IEEEtempbiographybox}}}%
% now place the author name and begin the bio text
\noindent\textbf{#2\ }\@IEEEgobbleleadPARNLSP}{\relax\let\par=\@IEEEBIOORGparCMD\par%
% 7/2001 V1.5 detect when the biography text is shorter than the photo area
% and pad the unused area - preventing a collision from the next biography entry
% MDS
\ifnum \prevgraf <\@IEEEtrantmpcountA\relax% detect when the biography text is shorter than the photo
    \advance\@IEEEtrantmpcountA by -\prevgraf% calculate how many lines we need to pad
    \advance\@IEEEtrantmpcountA by -1\relax% we compensate for the fact that we indented an extra line
    \@IEEEtrantmpdimenA=\baselineskip% calculate the length of the padding
    \multiply\@IEEEtrantmpdimenA by \@IEEEtrantmpcountA%
    \noindent\rule{0pt}{\@IEEEtrantmpdimenA}% insert an invisible support strut
\fi%
\par\normalfont}



% V1.6
% added biography without a photo environment
\newenvironment{IEEEbiographynophoto}[1]{%
% Make an entry into the table of contents only if we have not done so before
\if@IEEEbiographyTOCentrynotmade%
% link labels to the biography counter so hyperref will jump
% to the biography, not the previous section
\setcounter{IEEEbiography}{-1}%
\refstepcounter{IEEEbiography}%
\addcontentsline{toc}{section}{Biographies}%
\global\@IEEEbiographyTOCentrynotmadefalse%
\fi%
% one more biography
\refstepcounter{IEEEbiography}%
% Make an entry for this name into the table of contents 
\addcontentsline{toc}{subsection}{#1}%
\normalfont\@IEEEcompsoconly{\sffamily}\footnotesize\interlinepenalty500%
\vskip 4\baselineskip plus 1fil minus 0\baselineskip%
\parskip=0pt\par%
\noindent\textbf{#1\ }\@IEEEgobbleleadPARNLSP}{\relax\par\normalfont}


% provide the user with some old font commands
% got this from article.cls
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}


% SPECIAL PAPER NOTICE COMMANDS
% 
% holds the special notice text
\def\@IEEEspecialpapernotice{\relax}
 
% for special papers, like invited papers, the user can do:
% \IEEEspecialpapernotice{(Invited Paper)} before \maketitle
\def\IEEEspecialpapernotice#1{\ifCLASSOPTIONconference%
\def\@IEEEspecialpapernotice{{\sublargesize\textit{#1}\vspace*{1em}}}%
\else%
\def\@IEEEspecialpapernotice{{\\*[1.5ex]\sublargesize\textit{#1}}\vspace*{-2ex}}%
\fi}




% PUBLISHER ID COMMANDS
% to insert a publisher's ID footer
% V1.6 \IEEEpubid has been changed so that the change in page size and style
% occurs in \maketitle. \IEEEpubid must now be issued prior to \maketitle
% use \IEEEpubidadjcol as before - in the second column of the title page
% These changes allow \maketitle to take the reduced page height into
% consideration when dynamically setting the space between the author 
% names and the maintext.
%
% the amount the main text is pulled up to make room for the
% publisher's ID footer
% The IEEE uses about 1.3\baselineskip for journals, 
% dynamic title spacing will clean up the fraction
\def\@IEEEpubidpullup{1.3\baselineskip}
\ifCLASSOPTIONtechnote
% for technotes it must be an integer of baselineskip as there can be no
% dynamic title spacing for two column mode technotes (the title is in the
% in first column) and we should maintain an integer number of lines in the
% second column
% There are some examples (such as older issues of "Transactions on
% Information Theory") in which the IEEE really pulls the text off the ID for
% technotes - about 0.55in (or 4\baselineskip). We'll use 2\baselineskip
% and call it even.
\def\@IEEEpubidpullup{2\baselineskip}
\fi

% V1.7 compsoc does not use a pullup
\ifCLASSOPTIONcompsoc
\def\@IEEEpubidpullup{0pt}
\fi

% holds the ID text
\def\@IEEEpubid{\relax}

% flag so \maketitle can tell if \IEEEpubid was called
\newif\if@IEEEusingpubid
\global\@IEEEusingpubidfalse
% issue this command in the page to have the ID at the bottom
% V1.6 use before \maketitle
\def\IEEEpubid#1{\def\@IEEEpubid{#1}\global\@IEEEusingpubidtrue}


% command which will pull up (shorten) the column it is executed in
% to make room for the publisher ID. Place in the second column of
% the title page when using \IEEEpubid
% Is smart enough not to do anything when in single column text or
% if the user hasn't called \IEEEpubid
% currently needed in for the second column of a page with the
% publisher ID. If not needed in future releases, please provide this
% command and define it as \relax for backward compatibility
% v1.6b do not allow command to operate if the peer review option has been 
% selected because \IEEEpubidadjcol will not be on the cover page.
% V1.7 do nothing if compsoc
\def\IEEEpubidadjcol{\ifCLASSOPTIONcompsoc\else\ifCLASSOPTIONpeerreview\else
\if@twocolumn\if@IEEEusingpubid\enlargethispage{-\@IEEEpubidpullup}\fi\fi\fi\fi}

% Special thanks to Peter Wilson, Daniel Luecking, and the other
% gurus at comp.text.tex, for helping me to understand how best to
% implement the IEEEpubid command in LaTeX.



%% Lockout some commands under various conditions

% general purpose bit bucket
\newsavebox{\@IEEEtranrubishbin}

% flags to prevent multiple warning messages
\newif\if@IEEEWARNthanks
\newif\if@IEEEWARNIEEEPARstart
\newif\if@IEEEWARNIEEEbiography
\newif\if@IEEEWARNIEEEbiographynophoto
\newif\if@IEEEWARNIEEEpubid
\newif\if@IEEEWARNIEEEpubidadjcol
\newif\if@IEEEWARNIEEEmembership
\newif\if@IEEEWARNIEEEaftertitletext
\@IEEEWARNthankstrue
\@IEEEWARNIEEEPARstarttrue
\@IEEEWARNIEEEbiographytrue
\@IEEEWARNIEEEbiographynophototrue
\@IEEEWARNIEEEpubidtrue
\@IEEEWARNIEEEpubidadjcoltrue
\@IEEEWARNIEEEmembershiptrue
\@IEEEWARNIEEEaftertitletexttrue


%% Lockout some commands when in various modes, but allow them to be restored if needed
%%
% save commands which might be locked out
% so that the user can later restore them if needed
\let\@IEEESAVECMDthanks\thanks
\let\@IEEESAVECMDIEEEPARstart\IEEEPARstart
\let\@IEEESAVECMDIEEEbiography\IEEEbiography
\let\@IEEESAVECMDendIEEEbiography\endIEEEbiography
\let\@IEEESAVECMDIEEEbiographynophoto\IEEEbiographynophoto
\let\@IEEESAVECMDendIEEEbiographynophoto\endIEEEbiographynophoto
\let\@IEEESAVECMDIEEEpubid\IEEEpubid
\let\@IEEESAVECMDIEEEpubidadjcol\IEEEpubidadjcol
\let\@IEEESAVECMDIEEEmembership\IEEEmembership
\let\@IEEESAVECMDIEEEaftertitletext\IEEEaftertitletext


% disable \IEEEPARstart when in draft mode
% This may have originally been done because the pre-V1.6 drop letter
% algorithm had problems with a non-unity baselinestretch
% At any rate, it seems too formal to have a drop letter in a draft
% paper.
\ifCLASSOPTIONdraftcls
\def\IEEEPARstart#1#2{#1#2\if@IEEEWARNIEEEPARstart\typeout{** ATTENTION: \noexpand\IEEEPARstart
 is disabled in draft mode (line \the\inputlineno).}\fi\global\@IEEEWARNIEEEPARstartfalse}
\fi
% and for technotes
\ifCLASSOPTIONtechnote
\def\IEEEPARstart#1#2{#1#2\if@IEEEWARNIEEEPARstart\typeout{** WARNING: \noexpand\IEEEPARstart
 is locked out for technotes (line \the\inputlineno).}\fi\global\@IEEEWARNIEEEPARstartfalse}
\fi


% lockout unneeded commands when in conference mode
\ifCLASSOPTIONconference
% when locked out, \thanks, \IEEEbiography, \IEEEbiographynophoto, \IEEEpubid,
% \IEEEmembership and \IEEEaftertitletext will all swallow their given text. 
% \IEEEPARstart will output a normal character instead
% warn the user about these commands only once to prevent the console screen
% from filling up with redundant messages
\def\thanks#1{\if@IEEEWARNthanks\typeout{** WARNING: \noexpand\thanks
 is locked out when in conference mode (line \the\inputlineno).}\fi\global\@IEEEWARNthanksfalse}
\def\IEEEPARstart#1#2{#1#2\if@IEEEWARNIEEEPARstart\typeout{** WARNING: \noexpand\IEEEPARstart
 is locked out when in conference mode (line \the\inputlineno).}\fi\global\@IEEEWARNIEEEPARstartfalse}


% LaTeX treats environments and commands with optional arguments differently.
% the actual ("internal") command is stored as \\commandname 
% (accessed via \csname\string\commandname\endcsname )
% the "external" command \commandname is a macro with code to determine
% whether or not the optional argument is presented and to provide the 
% default if it is absent. So, in order to save and restore such a command
% we would have to save and restore \\commandname as well. But, if LaTeX
% ever changes the way it names the internal names, the trick would break.
% Instead let us just define a new environment so that the internal
% name can be left undisturbed.
\newenvironment{@IEEEbogusbiography}[2][]{\if@IEEEWARNIEEEbiography\typeout{** WARNING: \noexpand\IEEEbiography
 is locked out when in conference mode (line \the\inputlineno).}\fi\global\@IEEEWARNIEEEbiographyfalse%
\setbox\@IEEEtranrubishbin\vbox\bgroup}{\egroup\relax}
% and make biography point to our bogus biography
\let\IEEEbiography=\@IEEEbogusbiography
\let\endIEEEbiography=\end@IEEEbogusbiography

\renewenvironment{IEEEbiographynophoto}[1]{\if@IEEEWARNIEEEbiographynophoto\typeout{** WARNING: \noexpand\IEEEbiographynophoto
 is locked out when in conference mode (line \the\inputlineno).}\fi\global\@IEEEWARNIEEEbiographynophotofalse%
\setbox\@IEEEtranrubishbin\vbox\bgroup}{\egroup\relax}

\def\IEEEpubid#1{\if@IEEEWARNIEEEpubid\typeout{** WARNING: \noexpand\IEEEpubid 
 is locked out when in conference mode (line \the\inputlineno).}\fi\global\@IEEEWARNIEEEpubidfalse}
\def\IEEEpubidadjcol{\if@IEEEWARNIEEEpubidadjcol\typeout{** WARNING: \noexpand\IEEEpubidadjcol
 is locked out when in conference mode (line \the\inputlineno).}\fi\global\@IEEEWARNIEEEpubidadjcolfalse}
\def\IEEEmembership#1{\if@IEEEWARNIEEEmembership\typeout{** WARNING: \noexpand\IEEEmembership
 is locked out when in conference mode (line \the\inputlineno).}\fi\global\@IEEEWARNIEEEmembershipfalse}
\def\IEEEaftertitletext#1{\if@IEEEWARNIEEEaftertitletext\typeout{** WARNING: \noexpand\IEEEaftertitletext
 is locked out when in conference mode (line \the\inputlineno).}\fi\global\@IEEEWARNIEEEaftertitletextfalse}
\fi


% provide a way to restore the commands that are locked out
\def\IEEEoverridecommandlockouts{%
\typeout{** ATTENTION: Overriding command lockouts (line \the\inputlineno).}%
\let\thanks\@IEEESAVECMDthanks%
\let\IEEEPARstart\@IEEESAVECMDIEEEPARstart%
\let\IEEEbiography\@IEEESAVECMDIEEEbiography%
\let\endIEEEbiography\@IEEESAVECMDendIEEEbiography%
\let\IEEEbiographynophoto\@IEEESAVECMDIEEEbiographynophoto%
\let\endIEEEbiographynophoto\@IEEESAVECMDendIEEEbiographynophoto%
\let\IEEEpubid\@IEEESAVECMDIEEEpubid%
\let\IEEEpubidadjcol\@IEEESAVECMDIEEEpubidadjcol%
\let\IEEEmembership\@IEEESAVECMDIEEEmembership%
\let\IEEEaftertitletext\@IEEESAVECMDIEEEaftertitletext}



% need a backslash character for typeout output
{\catcode`\|=0 \catcode`\\=12
|xdef|@IEEEbackslash{\}}


% hook to allow easy disabling of all legacy warnings
\def\@IEEElegacywarn#1#2{\typeout{** ATTENTION: \@IEEEbackslash #1 is deprecated (line \the\inputlineno).
Use \@IEEEbackslash #2 instead.}}


% provide some legacy IEEEtran commands
\def\IEEEcompsoctitleabstractindextext{\@IEEElegacywarn{IEEEcompsoctitleabstractindextext}{IEEEtitleabstractindextext}\IEEEtitleabstractindextext}
\def\IEEEdisplaynotcompsoctitleabstractindextext{\@IEEElegacywarn{IEEEdisplaynotcompsoctitleabstractindextext}{IEEEdisplaynontitleabstractindextext}\IEEEdisplaynontitleabstractindextext}
% provide some legacy IEEEtran environments


% V1.8a no more support for these legacy commands
%\def\authorblockA{\@IEEElegacywarn{authorblockA}{IEEEauthorblockA}\IEEEauthorblockA}
%\def\authorblockN{\@IEEElegacywarn{authorblockN}{IEEEauthorblockN}\IEEEauthorblockN}
%\def\authorrefmark{\@IEEElegacywarn{authorrefmark}{IEEEauthorrefmark}\IEEEauthorrefmark}
%\def\PARstart{\@IEEElegacywarn{PARstart}{IEEEPARstart}\IEEEPARstart}
%\def\pubid{\@IEEElegacywarn{pubid}{IEEEpubid}\IEEEpubid}
%\def\pubidadjcol{\@IEEElegacywarn{pubidadjcol}{IEEEpubidadjcol}\IEEEpubidadjcol}
%\def\specialpapernotice{\@IEEElegacywarn{specialpapernotice}{IEEEspecialpapernotice}\IEEEspecialpapernotice}
% and environments
%\def\keywords{\@IEEElegacywarn{keywords}{IEEEkeywords}\IEEEkeywords}
%\def\endkeywords{\endIEEEkeywords}
% V1.8 no more support for legacy IED list commands
%\let\labelindent\IEEElabelindent
%\def\calcleftmargin{\@IEEElegacywarn{calcleftmargin}{IEEEcalcleftmargin}\IEEEcalcleftmargin}
%\def\setlabelwidth{\@IEEElegacywarn{setlabelwidth}{IEEEsetlabelwidth}\IEEEsetlabelwidth}
%\def\usemathlabelsep{\@IEEElegacywarn{usemathlabelsep}{IEEEusemathlabelsep}\IEEEusemathlabelsep}
%\def\iedlabeljustifyc{\@IEEElegacywarn{iedlabeljustifyc}{IEEEiedlabeljustifyc}\IEEEiedlabeljustifyc}
%\def\iedlabeljustifyl{\@IEEElegacywarn{iedlabeljustifyl}{IEEEiedlabeljustifyl}\IEEEiedlabeljustifyl}
%\def\iedlabeljustifyr{\@IEEElegacywarn{iedlabeljustifyr}{IEEEiedlabeljustifyr}\IEEEiedlabeljustifyr}
% V1.8 no more support for QED and proof stuff
%\def\QED{\@IEEElegacywarn{QED}{IEEEQED}\IEEEQED}
%\def\QEDclosed{\@IEEElegacywarn{QEDclosed}{IEEEQEDclosed}\IEEEQEDclosed}
%\def\QEDopen{\@IEEElegacywarn{QEDopen}{IEEEQEDopen}\IEEEQEDopen}
%\AtBeginDocument{\def\proof{\@IEEElegacywarn{proof}{IEEEproof}\IEEEproof}\def\endproof{\endIEEEproof}}
% V1.8 no longer support biography or biographynophoto
%\def\biography{\@IEEElegacywarn{biography}{IEEEbiography}\IEEEbiography}
%\def\biographynophoto{\@IEEElegacywarn{biographynophoto}{IEEEbiographynophoto}\IEEEbiographynophoto}
%\def\endbiography{\endIEEEbiography}
%\def\endbiographynophoto{\endIEEEbiographynophoto}
% V1.7 and later no longer supports \overrideIEEEmargins
%\def\overrideIEEEmargins{%
%\typeout{** WARNING: \string\overrideIEEEmargins \space no longer supported (line \the\inputlineno).}%
%\typeout{** Use the \string\CLASSINPUTinnersidemargin, \string\CLASSINPUToutersidemargin \space controls instead.}}

\endinput

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% End of IEEEtran.cls  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% That's all folks!

