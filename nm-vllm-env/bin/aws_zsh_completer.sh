# Source this file to activate auto completion for zsh using the bash
# compatibility helper.  Make sure to run `compinit` before, which should be
# given usually.
#
# % source /path/to/zsh_complete.sh
#
# Typically that would be called somewhere in your .zshrc.
#
# Note, the overwrite of _bash_complete() is to export COMP_LINE and COMP_POINT
# That is only required for zsh <= edab1d3dbe61da7efe5f1ac0e40444b2ec9b9570
#
# https://github.com/zsh-users/zsh/commit/edab1d3dbe61da7efe5f1ac0e40444b2ec9b9570
#
# zsh releases prior to that version do not export the required env variables!

autoload -Uz bashcompinit
bashcompinit -i

_bash_complete() {
  local ret=1
  local -a suf matches
  local -x COMP_POINT COMP_CWORD
  local -a COMP_WORDS COMPREPLY BASH_VERSINFO
  local -x COMP_LINE="$words"
  local -A savejobstates savejobtexts

  (( COMP_POINT = 1 + ${#${(j. .)words[1,CURRENT]}} + $#QIPREFIX + $#IPREFIX + $#PREFIX ))
  (( COMP_CWORD = CURRENT - 1))
  COMP_WORDS=( $words )
  BASH_VERSINFO=( 2 05b 0 1 release )

  savejobstates=( ${(kv)jobstates} )
  savejobtexts=( ${(kv)jobtexts} )

  [[ ${argv[${argv[(I)nospace]:-0}-1]} = -o ]] && suf=( -S '' )

  matches=( ${(f)"$(compgen $@ -- ${words[CURRENT]})"} )

  if [[ -n $matches ]]; then
    if [[ ${argv[${argv[(I)filenames]:-0}-1]} = -o ]]; then
      compset -P '*/' && matches=( ${matches##*/} )
      compset -S '/*' && matches=( ${matches%%/*} )
      compadd -Q -f "${suf[@]}" -a matches && ret=0
    else
      compadd -Q "${suf[@]}" -a matches && ret=0
    fi
  fi

  if (( ret )); then
    if [[ ${argv[${argv[(I)default]:-0}-1]} = -o ]]; then
      _default "${suf[@]}" && ret=0
    elif [[ ${argv[${argv[(I)dirnames]:-0}-1]} = -o ]]; then
      _directories "${suf[@]}" && ret=0
    fi
  fi

  return ret
}

complete -C aws_completer aws
