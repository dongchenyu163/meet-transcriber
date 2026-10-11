"""Align diarization turns to retained ASR word timestamps without losing text."""
from bisect import bisect_left, bisect_right


def speaker_label(line):
    if line.get('source') == 'mic':
        return '自己（麦克风）'
    speaker = line.get('speaker')
    if speaker == 'OVERLAP':
        return '多人同时说话'
    if speaker == 'UNKNOWN':
        return '未知说话人'
    if speaker:
        return line.get('speaker_name') or speaker.replace('SPEAKER_', '说话人 ')
    return '会议声音' if line.get('source') == 'system' else '语音'


def attribute(lines, turns, names=None, enabled=False):
    names = names or {}
    turns = sorted(turns, key=lambda t: t['start'])
    starts = [t['start'] for t in turns]
    max_ends = []
    maximum = float('-inf')
    for turn in turns:
        maximum = max(maximum, turn['end'])
        max_ends.append(maximum)

    def identify(word):
        candidates = turns[bisect_right(max_ends, word['start']):bisect_left(starts, word['end'])]
        scores, labels = {}, {}
        for turn in candidates:
            duration = min(word['end'], turn['end']) - max(word['start'], turn['start'])
            if duration > 0:
                key = turn['speaker']
                scores[key] = scores.get(key, 0) + duration
                labels[key] = names.get(key) or turn.get('speaker_name')
        for i, first in enumerate(candidates):
            for second in candidates[i + 1:]:
                overlap = min(word['end'], first['end'], second['end']) - max(word['start'], first['start'], second['start'])
                if first['speaker'] != second['speaker'] and overlap > .1 * max(.1, word['end'] - word['start']):
                    return 'OVERLAP', None
        if not scores:
            return ('UNKNOWN', None) if enabled else (None, None)
        ordered = sorted(scores, key=scores.get, reverse=True)
        best = ordered[0]
        if len(ordered) > 1 and scores[best] < scores[ordered[1]] * 1.2:
            return 'UNKNOWN', None
        return best, labels[best]

    output = []
    for original in lines:
        if original.get('source') != 'system' or not enabled:
            output.append(original.copy())
            continue
        words = original.get('words') or [{'start': original['start'], 'end': original['end'], 'text': original['text']}]
        groups = []
        for word in words:
            key = identify(word)
            if groups and groups[-1][0] == key:
                groups[-1][1].append(word)
            else:
                groups.append((key, [word]))
        for (speaker, name), group in groups:
            output.append({**original, 'start': group[0]['start'], 'end': group[-1]['end'],
                'text': ''.join(w['text'] for w in group).strip(), 'words': group,
                'speaker': speaker, 'speaker_name': name})
    return output
