"""Main entry point: python run_local_cuda.py"""
from limited_ct.local_run import main

if __name__=='__main__':
    try:
        main()
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError) as error:
        raise SystemExit(str(error)) from error
