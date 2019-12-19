import itertools
import json
import pickle


def read_json(loc):
    with open(loc) as file:
        loaded_dict = json.load(file)
    file.close()
    return loaded_dict


characters_test_dict = read_json('../../resources/characters_analysis/characters_test.json')


def read_txt(loc):
    with open(loc) as file:
        data = file.readlines()
    file.close()
    return data


characters_train = read_txt('../../resources/characters_analysis/characters_frequency_train.txt')
characters_val = read_txt('../../resources/characters_analysis/characters_frequency_val.txt')
characters_test = read_txt('../../resources/characters_analysis/characters_frequency_test.txt')

print(f'{len(characters_train)} train, {len(characters_val)} val, {len(characters_test)} test characters')

diverse_characters_ccs = set()
diverse_characters_vs = set()


def scorer(gt_characters, story, _id):
    good = 0
    for character in gt_characters:
        if character in story:
            good += 1
            if _id == 1:
                diverse_characters_vs.add(character)
            else:
                diverse_characters_ccs.add(character)

    return float(good) / float(len(gt_characters))


def evaluate(generated, _id):
    score = 0.0
    of_interest = 0
    sample_scores = []
    for sample in generated:
        story_id = sample['story_id'][0]
        story = sample['story']

        if story_id in characters_test_dict:
            of_interest += 1
            expected_characters = list(set(list(itertools.chain.from_iterable(characters_test_dict[story_id]))))
            score_now = scorer(expected_characters, story, _id)
            score += score_now
            sample_scores.append(score_now)

    print(f'qualified samples: {of_interest}, score: {score}, overall score: {score / float(of_interest)}')
    return sample_scores


def evaluate_diversity(generated, _id):
    for sample in generated:
        story = sample['story']
        # for character in characters_train:
        #     char = character.replace('\'', '').strip()[1:-1].split(', ')[0]
        #     if char in story:
        #         if _id == 1:
        #             diverse_characters_vs.add(char)
        #         else:
        #             diverse_characters_ccs.add(char)
        #
        # for character in characters_val:
        #     char = character.replace('\'', '').strip()[1:-1].split(', ')[0]
        #     if char in story:
        #         if _id == 1:
        #             diverse_characters_vs.add(char)
        #         else:
        #             diverse_characters_ccs.add(char)

        for character in characters_test:
            char = character.replace('\'', '').strip()[1:-1].split(', ')[0]
            if char in story:
                if _id == 1:
                    diverse_characters_vs.add(char)
                else:
                    diverse_characters_ccs.add(char)


quality_vs = {}
quality_ccs = {}


def evaluate_quality(generated, _id):
    for character in characters_test:
        char = character.replace('\'', '').strip()[1:-1].split(', ')[0]
        for sample in generated:
            story = sample['story'].split()
            if _id == 1:
                if char in quality_vs:
                    quality_vs[char] += story.count(char)
                else:
                    quality_vs[char] = story.count(char)
            else:
                if char in quality_ccs:
                    quality_ccs[char] += story.count(char)
                else:
                    quality_ccs[char] = story.count(char)
        # for character in characters_train:
        #     char = character.replace('\'', '').strip()[1:-1].split(', ')[0]
        #     if char in story:
        #         if _id == 1:
        #             diverse_characters_vs.add(char)
        #         else:
        #             diverse_characters_ccs.add(char)
        #
        # for character in characters_val:
        #     char = character.replace('\'', '').strip()[1:-1].split(', ')[0]
        #     if char in story:
        #         if _id == 1:
        #             diverse_characters_vs.add(char)
        #         else:
        #             diverse_characters_ccs.add(char)


if __name__ == '__main__':
    vs_generated_stories_test = read_json('../../resources/results/results_baseline_99.json')
    ccs_generated_stories_test = read_json('../../resources/results/results_baseline_cc_99.json')

    print('\n========== Evaluating baseline ==========')
    # scores_baseline = evaluate(vs_generated_stories_test, 1)
    evaluate_diversity(vs_generated_stories_test, 1)
    # evaluate_quality(vs_generated_stories_test, 1)
    # [i for i in scores_ccs if i != 0.0]

    # with open('./vs_quality.pkl', 'wb') as file:
    #     pickle.dump(quality_vs, file, pickle.HIGHEST_PROTOCOL)

    print('\n========== Evaluating CCS baseline ==========')
    # scores_ccs = evaluate(ccs_generated_stories_test, 2)
    evaluate_diversity(ccs_generated_stories_test, 2)
    # evaluate_quality(ccs_generated_stories_test, 2)

    # with open('./ccs_quality.pkl', 'wb') as file:
    #     pickle.dump(quality_ccs, file, pickle.HIGHEST_PROTOCOL)

    print('\n')
