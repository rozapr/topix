import math

DOC_KEY = 'd'
CLUSTER_KEY = 'c'
SUB_CLUSTER_KEY = 'sc'

def convert_to_topix_format(d):
    result = {}
    clusters = []
    documents = []
    
    doc_to_id = {}
    
    for c_i, cluster in enumerate(d):        
        cluster_obj = {
            'cluster_id' : f'{CLUSTER_KEY}_{c_i}',
            'cluster_name' : cluster['description'][0],
            'top_words' : [],
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
        
        # sub clusters = top words, docs distributed equally (not real clustering)
        sub_clusters = []
        cluster_size = len(cluster['documents'])
        sub_cluster_size = math.floor(cluster_size / (len(cluster['description']) - 1))
        
        for cs_i, word in enumerate(cluster['description'][1:]):
            size = sub_cluster_size
            cluster_size -= size
            if cluster_size < sub_cluster_size:
                size += cluster_size
            
            sub_cluster_obj = {
                'cluster_id' : f'{SUB_CLUSTER_KEY}_{c_i}_{cs_i}',
                'cluster_name' : word,
                'top_words' : [],
                'number_of_documents' : size,
                'list_of_documents_ids' : []
            }
            sub_clusters.append(sub_cluster_obj)
        
        cluster_obj['sub_clusters'] = sub_clusters
        
        clusters.append(cluster_obj)

    result['clusters'] = clusters
    result['num_of_clusters'] = len(clusters)
    result['documents'] = documents
    result['total_number_of_documents'] = len(doc_to_id)
    
    return result

