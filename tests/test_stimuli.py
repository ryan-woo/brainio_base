from brainio_base.stimuli import StimulusSet


class TestPreservation:
    def test_subselection(self):
        stimulus_set = StimulusSet([{'image_id': i, 'image_paths': f'/dummy/path/{i}'} for i in range(100)])
        stimulus_set = stimulus_set[stimulus_set['image_id'].isin(stimulus_set['image_id'].values[:3])]
        assert stimulus_set.get_image(stimulus_set['image_id'].values[0]) is not None
