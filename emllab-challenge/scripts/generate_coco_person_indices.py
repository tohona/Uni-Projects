from challenge.utils.dataloader import COCODataset, COCO_PERSON_CLASS_NUMBER
import json
import tqdm

def main():
    train_dataset = COCODataset(data_dir='./data/', train=True, do_transform=False)
    train_indices = find_person_indices(train_dataset)

    val_dataset = COCODataset(data_dir='./data/', train=False, do_transform=False)
    val_indices = find_person_indices(val_dataset)

    indices = {
        'train2017': train_indices,
        'val2017': val_indices
    }

    with open('./data/person_indices_coco.json', 'w') as file:
        json.dump(indices, file, indent=4)


def find_person_indices(dataset):
    #invalid_boxes = []
    indices = []
    for idx, (image, target) in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
        #print(idx)
        if any(item['category_id'] == COCO_PERSON_CLASS_NUMBER for item in target):
            indices.append(idx)
        #for item in target:
        #    box = item['bbox']
        #    if box[2] == 0 or box[3] == 0:
        #        print('invalid bbox: ', box)
        #        invalid_boxes.append((idx, box))
    #print(invalid_boxes)
    return indices



if __name__ == '__main__':
    main()