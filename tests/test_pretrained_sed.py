from nv_fix.pretrained_sed import DetectedEvent, event_intervals


def test_sed_events_become_merged_nv_intervals():
    events = [
        DetectedEvent("Laughter", 1.0, 2.0, 0.8),
        DetectedEvent("Human sounds", 1.5, 2.4, 0.6),
        DetectedEvent("Sigh", 4.0, 4.5, 0.9),
    ]
    assert event_intervals(events) == [(1.0, 2.4), (4.0, 4.5)]
