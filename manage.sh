#!/usrbin/env bash

COMMAND=$1
ARGUMENT=$2

## Interpret meaning of command line argument depending on which
## function will receive it.

case $COMMAND in
    run_tests_par | compile_and_test_par)     N_PROC=${ARGUMENT:-auto} ;;
    *)                                PYTHON_VERSION=${ARGUMENT}       ;;
esac

# If PYTHON_VERSION was not specified as an argument, deduce it from
# the conda environment

if [[ $PYTHON_VERSION = "" ]]; then
    PYTHON_VERSION=${CONDA_DEFAULT_ENV:3:3}
fi

function install_and_check {
    install
    run_tests
}

function install {
    install_conda
    make_environment
    python_version_env
    compile_cython_components
}

function install_conda {
    # Identifies your operating system and installs the appropriate
    # version of conda. We only consider Linux and OS X at present.

    # Does nothing if conda is already found in your PATH.

    case "$(uname -s)" in

        Darwin)
            export CONDA_OS=MacOSX
            ;;

        Linux)
            export CONDA_OS=Linux
            ;;

        # CYGWIN*|MINGW32*|MSYS*)
        #   echo 'MS Windows'
        #   ;;

        *)
            echo "Installation only supported on Linux and OS X"
            exit 1
            ;;
    esac

    if conda --version ; then
        echo Conda already installed. Skipping conda installation.
    else
        echo Installing conda for $CONDA_OS
        if which wget; then
            wget https://repo.continuum.io/miniconda/Miniconda3-4.5.4-${CONDA_OS}-x86_64.sh -O miniconda.sh
        else
            curl https://repo.continuum.io/miniconda/Miniconda3-4.5.4-${CONDA_OS}-x86_64.sh -o miniconda.sh
        fi
        bash miniconda.sh -b -p $HOME/miniconda
        CONDA_SH=$HOME/miniconda/etc/profile.d/conda.sh
        source $CONDA_SH
        echo Activated conda by sourcing $CONDA_SH
    fi
}

CONDA_ENV_TAG=2020-03-10
CONDA_ENV_NAME=IC-${PYTHON_VERSION}-${CONDA_ENV_TAG}

function make_environment {
    YML_FILENAME=environment-${CONDA_ENV_NAME}.yml

    echo creating ${YML_FILENAME}

    cat <<EOF > ${YML_FILENAME}
name: ${CONDA_ENV_NAME}
dependencies:
- python       = ${PYTHON_VERSION}
# *REMEMBER TO CHANGE CONDA_ENV_TAG WHEN CHANGING VERSION NUMBERS*
- cython       = 0.29.15
- jupyter      = 1.0.0
- jupyterlab   = 1.2.6
- matplotlib   = 3.1.3
- networkx     = 2.4
- notebook     = 6.0.3
- numpy        = 1.18.1
- pandas       = 1.0.1
- seaborn      = 0.10.0
- pymysql      = 0.9.2
- pytables     = 3.6.1
- pytest       = 5.3.5
- scipy        = 1.4.1
- sphinx       = 2.4.0
- tornado      = 6.0.3
- flaky        = 3.6.1
- coverage     = 5.0
- hypothesis   = 5.5.4
- pytest-xdist = 1.31.0
- pip          = 20.0.2
- pip :
  - pytest-instafail
EOF

    conda env create -f ${YML_FILENAME}
}

function make_environment_if_missing {
    if ! conda env list | grep ${CONDA_ENV_NAME}
    then
        make_environment
    fi
}

function switch_to_conda_env {
    conda deactivate # in case some other environment was already active
    make_environment_if_missing ${CONDA_ENV_NAME}
    conda activate ${CONDA_ENV_NAME}
}

function python_version_env {
    switch_to_conda_env
    # Set IC environment variables and download database
    ic_env
}

function work_in_python_version {
    work_in_python_version_no_tests
    run_tests_par
}

function work_in_python_version_no_tests {
    if ! which conda >> /dev/null
    then
       install_conda
    fi

    if ! (conda env list | grep ${CONDA_ENV_NAME}) >> /dev/null
    then
        make_environment
    fi

    python_version_env
    compile_cython_components
}

function ensure_environment_matches_checked_out_version {

    # Ensure that the currently active conda environment name contains
    # the tag that corresponds to this version of the code
    if [[ $CONDA_DEFAULT_ENV != *$CONDA_ENV_TAG ]]; then
        switch_to_conda_env
    fi

    # Ensure that the test database is present
    if [ ! -f invisible_cities/database/localdb.NEWDB.sqlite3 ]; then
        download_test_db
    fi
}

function run_tests {
    ensure_environment_matches_checked_out_version
    # Run the test suite
    pytest --instafail --no-success-flaky-report
}

function run_tests_par {
    ensure_environment_matches_checked_out_version
    # Run the test suite
    STATUS=0
    pytest --instafail -n ${N_PROC:-auto} -m "not serial" --no-success-flaky-report || STATUS=$?
    pytest --instafail                    -m      serial  --no-success-flaky-report || STATUS=$?
    [[ $STATUS = 0 ]]
}

function ic_env {
    export ICTDIR=`pwd`
    echo ICDIR set to $ICTDIR

    export ICDIR=$ICTDIR/invisible_cities/
    echo ICDIR set to $ ICDIR

    export PYTHONPATH=$ICTDIR
    echo PYTHONPATH set to $PYTHONPATH

    export NBDIR=`pwd`
    echo NBDIR set to $NBDIR

    export IC_NOTEBOOK_DIR=$NBDIR/invisible_cities/
    echo IC_NOTEBOOK_DIR set to $ IC_NOTEBOOK_DIR

    export PATH=$ICTDIR/bin:$PATH
    echo $ICTDIR/bin added to path
}

function show_ic_env {
    conda env list

    echo "ICTDIR set to $ICTDIR"
    echo "ICDIR  set to $ICDIR"
    echo PYTHONPATH set to $PYTHONPATH
}

function download_test_db {
    echo Downloading database
    python $ICDIR/database/download.py $ARGUMENT
}

function compile_cython_components {
    python setup.py develop
}

function compile_and_test {
    compile_cython_components
    run_tests
}

function compile_and_test_par {
    compile_cython_components
    run_tests_par
}

function clean {
    echo "Cleaning IC generated files:"
    FILETYPES='*.c *.so *.pyc __pycache__'
    for TYPE in $FILETYPES
    do
                echo Cleaning $TYPE files
        REMOVE=`find . -name $TYPE`
        if [ ! -z "${REMOVE// }" ]
        then
            for FILE in $REMOVE
            do
               rm -rf $FILE
            done
        else
            echo Nothing found to clean in $TYPE
        fi
    done
}


THIS=manage.sh

## Main command dispatcher

case $COMMAND in
    install_and_check)               install_and_check ;;
    install)                         install ;;
    work_in_python_version)          work_in_python_version ;;
    work_in_python_version_no_tests) work_in_python_version_no_tests ;;
    make_environment)                make_environment ;;
    run_tests)                       run_tests ;;
    run_tests_par)                   run_tests_par ;;
    compile_and_test)                compile_and_test ;;
    compile_and_test_par)            compile_and_test_par ;;
    download_test_db)                download_test_db ;;
    clean)                           clean ;;
    show_ic_env)                     show_ic_env ;;

    *) echo Unrecognized command: ${COMMAND}
       echo
       echo Usage:
       echo
       echo "source $THIS install_and_check X.Y"
       echo "source $THIS install X.Y"
       echo "source $THIS work_in_python_version X.Y"
       echo "source $THIS work_in_python_version_no_tests X.Y"
       echo "source $THIS switch_to_conda_env X.Y"
       echo "bash   $THIS make_environment X.Y"
       echo "bash   $THIS run_tests"
       echo "bash   $THIS run_tests_par"
       echo "bash   $THIS compile_and_test"
       echo "bash   $THIS compile_and_test_par"
       echo "bash   $THIS download_test_db"
       echo "bash   $THIS clean"
       echo "bash   $THIS show_ic_env"
       ;;
esac
