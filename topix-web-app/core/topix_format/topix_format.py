from typing import List, Dict, Any
import math

DOC_KEY = 'd'
CLUSTER_KEY = 'c'
SUB_CLUSTER_KEY = 'sc'

def convert_to_topix_format(d : List[Dict[str, Any]]) -> Dict[str, Any]:
    result = {}
    clusters = []
    documents = []
    
    doc_to_id = {}
    
    for c_i, cluster in enumerate(d):        
        cluster_obj = {
            'cluster_id' : f'{CLUSTER_KEY}_{c_i}',
            'cluster_name' : ', '.join(cluster['description']),
            'top_words' : cluster['description'],
            'number_of_documents' : len(cluster['documents'])
        }
        
        # docs
        list_of_documents_ids = []
        for doc in cluster['documents']:
            if doc not in doc_to_id:
                doc_id = len(doc_to_id)
                doc_to_id[doc] = f'{DOC_KEY}_{doc_id}'
                
                doc_obj = {
                    'document_id' : doc_to_id[doc],
                    'header' : '',
                    'text' : doc,
                    'link' : ''
                }
                documents.append(doc_obj)
            
            doc_id = doc_to_id[doc]
            list_of_documents_ids.append(doc_id)
        
        cluster_obj['list_of_documents_ids'] = list_of_documents_ids
        
        # sub clusters
        sub_clusters = []
        if 'subtopics' not in cluster:
            # single sub cluster = cluster
            sub_cluster_obj = {
                'cluster_id' : f'{SUB_CLUSTER_KEY}_{c_i}_0',
                'cluster_name' : '',
                'top_words' : [],
                'number_of_documents' : len(list_of_documents_ids),
                'list_of_documents_ids' : list_of_documents_ids
            }
            sub_clusters.append(sub_cluster_obj)
        else:
            for sc_i, sub_cluster in enumerate(cluster['subtopics']):
                sub_cluster_obj = {
                    'cluster_id' : f'{SUB_CLUSTER_KEY}_{c_i}_{sc_i}',
                    'cluster_name' : ', '.join(sub_cluster['description']),
                    'top_words' : sub_cluster['description'],
                    'number_of_documents' : len(sub_cluster['documents']),
                    'list_of_documents_ids' : [doc_to_id[doc] for doc in sub_cluster['documents']]
                }
                sub_clusters.append(sub_cluster_obj)
        
        cluster_obj['sub_clusters'] = sub_clusters
        
        clusters.append(cluster_obj)

    result['clusters'] = clusters
    result['num_of_clusters'] = len(clusters)
    result['documents'] = documents
    result['total_number_of_documents'] = len(doc_to_id)
    
    return result

