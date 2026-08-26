# -*- coding: utf-8 -*-
"""一次文本任务（submit 的返回值），与后端无关。"""
import threading


class Job:
    """一次文本任务（submit 的返回值）。wait() 阻塞到该任务播完（或被打断取消）。"""

    __slots__ = ("job_id", "gen", "sentences", "n", "save_dir", "save_wav_path",
                 "t_start", "timing", "_event", "canceled")

    def __init__(self, job_id, gen, sentences, save_dir, save_wav_path, t_start):
        self.job_id = job_id
        self.gen = gen
        self.sentences = sentences
        self.n = len(sentences)
        self.save_dir = save_dir
        self.save_wav_path = save_wav_path
        self.t_start = t_start
        self.timing = []           # 逐句记录（profile 开时才有内容）
        self.canceled = False
        self._event = threading.Event()

    def wait(self):
        """阻塞到本任务播完或被取消，返回逐句时序记录（与 speak() 一致）。"""
        self._event.wait()
        return self.timing

    def mark_done(self, canceled=False):
        self.canceled = canceled
        self._event.set()
