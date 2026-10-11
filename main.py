"""Launch the PySide6 application; utilities remain available from the CLI."""
import argparse
import multiprocessing
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description='Meet Notes · 本地会议转录与 DeepSeek 纪要')
    parser.add_argument('command', nargs='?', default='gui', choices=['gui', 'devices', 'recover'])
    parser.add_argument('--session', help='recover 命令使用的会议目录')
    args = parser.parse_args(argv)
    if args.command == 'devices':
        from audio import list_devices
        for device in list_devices():
            print(f'{"会议播放声" if device.loopback else "麦克风"}: {device.name}\n  {device.id}')
        return 0
    if args.command == 'recover':
        if not args.session:
            parser.error('recover 需要 --session 指定会议目录')
        from storage import recover_exports
        recover_exports(args.session)
        print('已从事件记录恢复 TXT / JSON。完整音频重转录请使用界面。')
        return 0
    from gui import run_gui
    return run_gui()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    sys.exit(main())
