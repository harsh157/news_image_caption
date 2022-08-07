
import json
import logging

try:
    image_maps = {}
    with open('expt/goodnews/h6_faces_pointer/serialization/generations.jsonl') as newmodel, open('/home/jupyter/data/GoodNews/expt/goodnews/8_transformer_faces/serialization/generations.jsonl') as oldmodel:
        for new, old in zip(newmodel, oldmodel):
            newcap = json.loads(new)
            image_id_new = newcap['image_path'].split('/')[-1].split('.')[0]
            oldcap = json.loads(old)
            image_id_old = oldcap['image_path'].split('/')[-1].split('.')[0]

            if image_id_new not in image_maps:
                image_maps[image_id_new] = {}
            image_maps[image_id_new]['new'] = newcap['generation']
            if image_id_old not in image_maps:
                image_maps[image_id_old] = {}
            image_maps[image_id_old]['old'] = oldcap['generation']
            image_maps[image_id_old]['org'] = oldcap['caption']
            image_maps[image_id_old]['img'] = image_id_old
            image_maps[image_id_old]['url'] = oldcap['web_url']

    print(len(image_maps))
    for imid in image_maps:
        if image_maps[imid]['new'] != image_maps[imid]['old']:
            print('new', image_maps[imid]['new'])
            print('old', image_maps[imid]['old'])
            print('org', image_maps[imid]['org'])
            print('img', image_maps[imid]['img'])
            print('img', image_maps[imid]['url'])
            print('-------------------------')
except:
    logging.exception('err')
    pass
