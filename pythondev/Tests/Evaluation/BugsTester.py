from Tests.Constants import DELIVERED
from Tests.Utils.TestsUtils import run_cmd
from Tests.Utils.StringUtils import join
from Tests.VersionTester import get_args

import bugzilla
import os


def get_bugs_dict():
    b = bugzilla.Bugzilla(
        url="http://bugzilla-srv/bugzilla/rest.cgi", api_key="ULBFcwSIyuS7GpiN0YHofn0znzoQthQDqJQGrJj8")
    bugs_list = b.query(b.build_query(component='Algo', status='CONFIRMED'))
    bugs_dict = {}
    for bug in bugs_list:
        if bug.is_open:
            bugs_dict[bug.id] = bug
    return bugs_dict


def run_version_tester(folder, bugs_dict, prev_versions):
    base_cmd = f' ./Tests/VersionTester.py --silent --plot_ref --plot_all --force'
    for bug_id, bug in bugs_dict.items():
        setups = bug.cf_session_ids
        if len(setups):
            run_cmd(base_cmd +
                    f' -setups {setups}  -result_dir {folder} -version bug_{bug_id} -compare_to {join(prev_versions)}')


def main_bug_tester(args):
    os.chdir(os.path.join(DELIVERED, args.version, args.version))
    result_dir = os.path.join(DELIVERED, args.version, 'stats', 'bugs')
    run_version_tester(result_dir, get_bugs_dict(), args.compare_to)


if __name__ == '__main__':
    main_bug_tester(get_args())
