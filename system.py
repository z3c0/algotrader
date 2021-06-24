import threading as thr
import datetime as dt
import argparse


from const import Arguments, System


class ShellComponent:
    '''A class for handling command-line arguments'''

    def __init__(self):
        arg_parser = argparse.ArgumentParser(prog='algotrader')

        boolean = {'action': 'store_true'}

        arg_parser.add_argument('-p', '--print', **boolean,
                                help='display account status')

        data_args = (Arguments.Data.Download, Arguments.Data.Transform)
        backtest_args = (Arguments.Backtest.Optimize,)
        position_args = (Arguments.Position.Strategize,
                         Arguments.Position.Apply)
        test_args = (Arguments.Testing.Verbose, Arguments.Testing.Rollback)

        data = arg_parser.add_argument_group(Arguments.Data.title)
        for arg in data_args:
            data.add_argument(f'-{arg.alias}', f'--{arg.flag}',
                              help=arg.help_text, **boolean)

        backtest = arg_parser.add_argument_group(Arguments.Backtest.title)
        for arg in backtest_args:
            backtest.add_argument(f'-{arg.alias}', f'--{arg.flag}',
                                  help=arg.help_text, **boolean)

        position = arg_parser.add_argument_group(Arguments.Position.title)
        for arg in position_args:
            position.add_argument(f'-{arg.alias}', f'--{arg.flag}',
                                  help=arg.help_text, **boolean)

        test = arg_parser.add_argument_group(Arguments.Testing.title)
        for arg in test_args:
            test.add_argument(f'-{arg.alias}', f'--{arg.flag}',
                              help=arg.help_text, **boolean)

        strategies = arg_parser.add_argument_group('strategies')
        strategies.add_argument('-m', '--momentum', **boolean, default=True)
        strategies.add_argument('-S', '--sma', **boolean)
        strategies.add_argument('-M', '--mean-reversion', **boolean)
        strategies.add_argument('-L', '--linear-regression', **boolean)

        args, unknown = arg_parser.parse_known_args()
        self.args = args
        self.unknown_args = unknown


class LogComponent:
    '''A thread-safe class for logging info to stdout or a specified file'''

    def __init__(self, stdout=True, path=None):
        if not stdout and path is None:
            print('[-]: a path is required when stdout is False')
            stdout = True

        if stdout and path is None:

            def _print_wrapper(*values, **kwargs):
                print(*values, **kwargs)

        elif stdout and path is not None:

            def _print_wrapper(*values, **kwargs):
                with open(path, 'a') as log_file:
                    print(*values, **kwargs, file=log_file)
                print(*values, **kwargs)

        else:

            def _print_wrapper(*values, **kwargs):
                with open(path, 'a') as log_file:
                    print(*values, **kwargs, file=log_file)

        self._write_func = _print_wrapper

        self._is_enabled = True
        self._print_lock = thr.Lock()

    def message(self, text):
        if self._is_enabled:
            with self._print_lock:
                self._write_func(f'[{dt.datetime.now()}]: {text}')

    def disable(self):
        self._is_enabled = False


class Input:
    shell = ShellComponent()


class Output:
    stdout = Input.shell.args.verbose

    if Input.shell.args.rollback:
        log = LogComponent(stdout=True)
    else:
        log = LogComponent(stdout=stdout, path=System.log)


if Input.shell.unknown_args:
    unknown_args = ', '.join(*Input.shell.unknown_args)
    Output.log.message(f'unknown: {unknown_args}')
