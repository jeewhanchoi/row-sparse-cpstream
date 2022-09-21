#!/bin/bash

module load intel/19
source /packages/intel/19/linux/pkg_bin/compilervars.sh -arch intel64 -platform linux

function special_execute() { 
    [ "$1" -eq "-123" ] && echo "flagY" || echo "flagN"; 
    shift; 
    set -x; "$@"; set +x; 
  }

function get_path_to_tensor() {
    declare -A path_to_tensors=(
        ["flickr"]="~/hpctensor/flickr-4d.tns"
        ["chicago"]="~/hpctensor/chicago-crime-comm.tns"
        ["uber"]="~/hpctensor/uber.tns"
        ["deli"]="~/hpctensor/delicious-4d.tns"
        ["patents"]="~/hpctensor/patents.tns"
        ["lbnl"]="~/hpctensor/lbnl-network.tns"
        ["nips"]="~/hpctensor/nips.tns"
        ["patents"]="~/hpctensor/patents.tns"
        ["lanl"]="~/hpctensor/lanl_one.tns"
    )
    echo "${path_to_tensors[${1}]}"
}

function get_streaming_mode() {
  declare -A streaming_mode=(
    ["uber"]="1"
    ["chicago"]="1"
    ["flickr"]="4" 
    ["deli"]="4" 
    ["patents"]="1" 
    ["lbnl"]="5" 
    ["nips"]="4" 
    ["lanl"]="1"
  )
  echo "${streaming_mode[${1}]}"
}

function get_reg() {
    declare -A reg_type=(
        ["uber"]="--reg=frob,1e-12,1,2,3,4"
        ["chicago"]="--reg=frob,1e-12,1,2,3,4"
        ["flickr"]="--reg=frob,1e-12,1,2,3,4"
        ["deli"]="--reg=frob,1e-12,1,2,3,4"
        ["patents"]="--reg=frob,1e-12,1,2,3"
        ["lbnl"]="--reg=frob,1e-12,1,2,3,4,5"
        ["nips"]="--reg=frob,1e-12,1,2,3,4"
        ["lanl"]="--reg=frob,1e-12,1,2,3,4,5"
    )
    echo "${reg_type[${1}]}"
}

function get_rsp() {
    [[ -z "$1" ]] && echo "" || echo "--use_rsp=True"
}

cmd="./build/Linux-x86_64/bin/splatt cpd -v --tol 5e-2 -r 16 --seed 23 --stream=$(get_streaming_mode $1) -t 56 $(get_rsp $2) $(get_reg $1) $(get_path_to_tensor $1)"
make && echo $cmd && eval "$cmd"

cmd="./build/Linux-x86_64/bin/splatt cpd -v --tol 5e-2 -r 16 --seed 33 --stream=$(get_streaming_mode $1) -t 56 $(get_rsp $2) $(get_reg $1) $(get_path_to_tensor $1)"
make && echo $cmd && eval "$cmd"

cmd="./build/Linux-x86_64/bin/splatt cpd -v --tol 5e-2 -r 16 --seed 45 --stream=$(get_streaming_mode $1) -t 56 $(get_rsp $2) $(get_reg $1) $(get_path_to_tensor $1)"
make && echo $cmd && eval "$cmd"
