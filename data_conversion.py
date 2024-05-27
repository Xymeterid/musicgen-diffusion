import pretty_midi

TIME_SHIFT_RESOLUTION = 0.002
NOTE_ON_FLAG = 0
NOTE_OFF_FLAG = 128
TIME_SHIFT_FLAG = 256

# Equal to 4 bars
SEQUENCE_DURATION = 10000


def midi_to_event_sequences(midi_file_path):
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)
    events = []
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            events.append(('note_on', note.start, note.pitch))
            events.append(('note_off', note.end, note.pitch))
    events.sort(key=lambda x: (x[1], x[0] != 'note_on'))

    last_time = 0
    event_sequences = []
    event_sequence = []
    for event in events:
        event_time = event[1]
        time_shift = event_time - last_time

        if time_shift > 0:
            time_shift_steps = int(time_shift // TIME_SHIFT_RESOLUTION)
            for _ in range(time_shift_steps):
                event_sequence.append(('time_shift', TIME_SHIFT_RESOLUTION))

                if len(event_sequence) >= SEQUENCE_DURATION:
                    event_sequences.append(event_sequence)
                    event_sequence = []

        event_sequence.append(event)
        last_time = event_time

        if len(event_sequence) >= SEQUENCE_DURATION:
            event_sequences.append(event_sequence)
            event_sequence = []

    return event_sequences


def events_to_token_sequence(event_sequence):
    return [tokenize_event(event) for event in event_sequence]


def tokenize_event(event):
    symbol = 0

    if event[0] == 'note_on':
        symbol = NOTE_ON_FLAG + event[2]
    elif event[0] == 'note_off':
        symbol = NOTE_OFF_FLAG + event[2]
    elif event[0] == 'time_shift':
        symbol = TIME_SHIFT_FLAG

    return symbol / TIME_SHIFT_FLAG


VELOCITY_BASE = 64
MAX_PLAYED_DURATION = 5000


def event_sequence_to_midi(event_sequence, output_midi_file_path):
    midi_data = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    current_time = 0
    ongoing_notes = {}

    for event in event_sequence:
        event_type = event[0]

        if event_type == 'time_shift':
            current_time += event[1]

            for note_pitch in ongoing_notes.keys():
                note_start_time, velocity, duration = ongoing_notes[note_pitch]
                duration += 1
                if duration >= MAX_PLAYED_DURATION:
                    end_time = note_start_time + (duration * TIME_SHIFT_RESOLUTION)
                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=note_pitch,
                        start=note_start_time,
                        end=end_time
                    )
                    instrument.notes.append(note)
                    del ongoing_notes[note_pitch]
                else:
                    ongoing_notes[note_pitch] = (note_start_time, velocity, duration)

        elif event_type == 'note_on':
            note_pitch = event[1]
            ongoing_notes[note_pitch] = (current_time, VELOCITY_BASE, 0)

        elif event_type == 'note_off':
            note_pitch = event[1]
            if note_pitch in ongoing_notes:
                start_time, velocity, _ = ongoing_notes.pop(note_pitch)
                end_time = current_time
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=note_pitch,
                    start=start_time,
                    end=end_time
                )
                instrument.notes.append(note)

    for note_pitch, (start_time, velocity, duration) in ongoing_notes.items():
        end_time = start_time + (duration * TIME_SHIFT_RESOLUTION)
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=note_pitch,
            start=start_time,
            end=end_time
        )
        instrument.notes.append(note)

    midi_data.instruments.append(instrument)
    midi_data.write(output_midi_file_path)


def token_sequence_to_events(token_sequence):
    return [detokenize_event(token) for token in token_sequence]


def detokenize_event(token):
    token = round(token * TIME_SHIFT_FLAG)

    if NOTE_ON_FLAG <= token < NOTE_OFF_FLAG:
        return ['note_on', token - NOTE_ON_FLAG]
    elif NOTE_OFF_FLAG <= token < TIME_SHIFT_FLAG:
        return ['note_off', token - NOTE_OFF_FLAG]
    elif TIME_SHIFT_FLAG <= token:
        time_shift = TIME_SHIFT_RESOLUTION
        return ['time_shift', time_shift]
