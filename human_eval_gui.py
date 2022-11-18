import PySimpleGUI as sg
from pathlib import Path
from mturk_process import MTurkProcessJSON
from mturk_statistics import MTurkStatistics
from mturk_self_replication import MturkSelfRepicaltion
import pandas as pd
import sys
import tkinter as tk
from tkinter import ttk
import traceback
import platform, os



class HumanEvalGUI:

    def __init__(self) -> None:
        self.theme = "Material1"
        self.font = "Arial 18"
        self.min_w = 800
        self.min_h = 500
        self.window = None

    @staticmethod
    def event_process_json(file, opendir=False):
        try:
            process_js = MTurkProcessJSON(file)
            process_js.process()
            process_js.savefiles()
            if opendir:
                dpath = process_js.savedir
                print(f"open diretory {dpath}")
                HumanEvalGUI.open_dir(process_js.savedir)
            return {"success": True, "error": None}
        except Exception:
            error_message = traceback.format_exc()
            return {"success": False, "error": error_message}

    @staticmethod
    def event_process_result_directory(dpath, opendir=False):
        try:
            mturk_stats = MTurkStatistics(dpath)
            mturk_stats.process()
            mturk_stats.savefiles()
            if opendir:
                dpath = mturk_stats.savedir
                print(f"open diretory {dpath}")
                HumanEvalGUI.open_dir(mturk_stats.savedir)
            return {"success": True, "error": None}
        except Exception:
            error_message = traceback.format_exc()
            return {"success": False, "error": error_message}

    @staticmethod
    def event_process_r1r2_directories(dpath_r1, dpath_r2, opendir=False):
        try:
            mturk_self_rep = MturkSelfRepicaltion(dpath_r1,dpath_r2)
            mturk_self_rep.process()
            mturk_self_rep.savefiles()
            if opendir:
                dpath = mturk_self_rep.savedir
                print(f"open diretory {dpath}")
                HumanEvalGUI.open_dir(mturk_self_rep.savedir)
            return {"success": True, "error": None}
        except Exception:
            error_message = traceback.format_exc()
            return {"success": False, "error": error_message}



    @staticmethod
    def open_dir(dpath):
        system_os = platform.system()
        if system_os == "Darwin":
            os.system(f'open "{dpath}"')


    def padding(self, n=1):
        return sg.Text("", font=f"Arial {n}"),

    def choose_btn(self, key):
        return sg.Button(button_text="choose", font=self.font, size=(6, 1), key=key)

    def text_input(self, key):
        return sg.Input(font=self.font, expand_x=True, background_color='white', size=(3, 1), key=key)

    def front_text(self, text):
        return sg.Text(text, font=self.font)

    def radio_btn(self, text, key, value=False, group="RADIOGROUP1"):
        return sg.Radio(text, group, font=self.font, default=value, key=key)

    def notification_text(self, text):
        return [
            sg.Push(),
            sg.Text(text, font=self.font),
            sg.Push(),
        ]

    def head(self, text, key, value):
        return [
            self.radio_btn(text, value, key),
            sg.Push(),
        ]

    def body(self, text, txt_key, btn_key):
        return [
            sg.Text(text, font=self.font),
            self.text_input(txt_key),
            self.choose_btn(btn_key),
        ]

    def file_picker(self):
        return sg.popup_get_file('', modal=False, initial_folder='./', no_window=True, multiple_files=False)

    def directory_picker(self):
        return sg.popup_get_folder('', modal=False, initial_folder='./', no_window=True)

    def error_popup(self, err):
        return sg.popup_error(err, title="Error Message", modal=True)


    def init_window(self):
        print("init window")
        w0, h0 = sg.Window.get_screen_size()
        w1 = int(w0 / 2 + 50)
        h1 = int(h0 / 2 + 50)
        w2, h2 = self.min_w, self.min_h
        w = max(w1, w2)
        h = max(h1, h2)
        layout = [
            [sg.Checkbox('Open directory when processing is completed', font="Arial 16", default=True,
                         key="-open-dir-", )],
            [self.padding(4)],
            [sg.Push(), sg.Text("", font=self.font, key='-notification-main-'), sg.Push()],
            # [sg.Push(), sg.Text("", font=self.font, key='-notification-text-'), sg.Push()],
            [self.padding(4)],

            # 1 process json

            [*self.head('Process the JSON file', True, '-radio-json-')],
            [*self.body('Path:', '-fpath-json-', '-choose-json-')],
            [self.padding(2)],

            # 2 process directory

            [*self.head(R"Process the 'Results/' directory", False, '-radio-results-dir-')],
            [*self.body('Path:', '-dpath-results-', '-choose-results-dir-')],
            [self.padding(2)],

            # 3 self-replication

            [*self.head(R"Process the 'Results/' directories of two runs for self-replication experiment", False,
                        '-radio-self-replication-')],
            [*self.body('Run1: ', '-dpath-r1-', '-choose-r1-dir-')],
            [*self.body('Run2: ', '-dpath-r2-', '-choose-r2-dir-')],

            [self.padding(6)],
            [sg.Button(button_text="PROCESS", font=self.font, expand_x=True, key='-process-')],
            [self.padding(1)],
            [sg.Button(button_text="RESET", font=self.font, expand_x=True, key='-reset-')],

            # ending
            [sg.VPush()],  # v padding
            [sg.Button(button_text="CLOSE", font=self.font, expand_x=True, key='-close-',
                       button_color=('black', "#eb5266"))]
        ]

        window = sg.Window(
            'Human Evaluation for Open-domain Dialogue Systems',
            layout, finalize=True,
            size=(w, h),
            element_justification='c',
            keep_on_top=True,
        )
        window.keep_on_top_clear()
        print(f"Window size: ({w}, {h})")
        self.window = window

    def start_gui(self):
        self.init_window()
        window = self.window

        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event == '-close-':
                break

            if event == "-choose-json-":
                path_text = self.file_picker()
                if path_text:
                    window.Element('-radio-json-').Update(value=True)
                    window['-fpath-json-'].Update(path_text)
            if event == "-choose-results-dir-":
                path_text = self.directory_picker()
                if path_text:
                    window.Element('-radio-results-dir-').Update(value=True)
                    window['-dpath-results-'].Update(path_text)
            if event in ['-choose-r1-dir-', '-choose-r2-dir-']:
                path_text = self.directory_picker()
                if path_text:
                    window.Element('-radio-self-replication-').Update(value=True)
                    if event == '-choose-r1-dir-':
                        window['-dpath-r1-'].Update(path_text)
                    elif event == '-choose-r2-dir-':
                        window['-dpath-r2-'].Update(path_text)

            if event == "-process-":
                window.Element('-notification-main-').Update(value="")
                radio_json = values['-radio-json-']
                radio_results_dir = values['-radio-results-dir-']
                radio_self_replication = values['-radio-self-replication-']
                opendir = values['-open-dir-']

                if radio_json:
                    fpath = values['-fpath-json-']
                    if fpath:
                        results = HumanEvalGUI.event_process_json(fpath, opendir)
                        if results['success']:
                            window.Element('-notification-main-').Update(value="Processing completed")
                        elif not results['success']:
                            error_message = results.get('error', 'unknow error, please check the code')
                            self.error_popup(error_message)
                            window.Element('-notification-main-').Update(value="Error occurs, please re-try")
                        window['-fpath-json-'].Update("")
                    else:
                        error_message = "Empty path: JSON file"
                        window.Element('-notification-main-').Update(value=error_message)
                elif radio_results_dir:
                    dpath = values['-dpath-results-']
                    if dpath:
                        results = HumanEvalGUI.event_process_result_directory(dpath, opendir)
                        if results['success']:
                            window.Element('-notification-main-').Update(value="Processing completed", visible=True)
                        elif not results['success']:
                            error_message = results.get('error', 'unknow error, please check the code')
                            self.error_popup(error_message)
                            window.Element('-notification-main-').Update(value="Error occurs, please re-try")
                        window['-dpath-results-'].Update("")
                    else:
                        error_message = "Empty path: 'Results/' directory"
                        window.Element('-notification-main-').Update(value=error_message)
                elif radio_self_replication:
                    # window.Element('-notification-main-').Update(value="33sxvasas"*100, visible=True)
                    dpath_r1 = values['-dpath-r1-']
                    dpath_r2 = values['-dpath-r2-']
                    if dpath_r1 and dpath_r2:
                        results = HumanEvalGUI.event_process_r1r2_directories(dpath_r1, dpath_r2, opendir)
                        if results['success']:
                            window.Element('-notification-main-').Update(value="Processing completed", visible=True)
                        elif not results['success']:
                            error_message = results.get('error', 'unknow error, please check the code')
                            self.error_popup(error_message)
                            window.Element('-notification-main-').Update(value="Error occurs, please re-try")
                    else:
                        error_message = "Empty path: 'Results/' directory of Run 1 or Run 2"
                        window.Element('-notification-main-').Update(value=error_message)
            if event == "-reset-":
                window.Element('-radio-json-').Update(value=True)
                window.Element('-open-dir-').Update(value=True)
                window.Element('-fpath-json-').Update(value="")
                window.Element('-dpath-results-').Update(value="")
                window.Element('-dpath-r1-').Update(value="")
                window.Element('-dpath-r2-').Update(value="")
                window.Element('-notification-main-').Update(value="")

        window.close()
        print("close window")


if __name__ == '__main__':
    pass
