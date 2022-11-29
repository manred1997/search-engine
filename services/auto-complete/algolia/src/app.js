// Replace with your own values
const searchClient = algoliasearch(
    'IPCD9865X5',
    '6fc947ffa33b8ef5ec4865891fe56fe7' // search only API key, not admin API key
    );
  
const search = instantsearch({
            indexName: 'autocomplete_18_4_2022_symptoms',
            searchClient,
            routing: true,
        });

// search.addWidgets([
//         instantsearch.widgets.configure({
//             hitsPerPage: 10,
//             })
//         ]);

search.addWidgets([
        instantsearch.widgets.searchBox({
            container: '#search-box',
            placeholder: 'Search for contacts',
            })
        ]);

search.addWidgets([
        instantsearch.widgets.hits({
            container: '#hits',
            templates: {
            item: document.getElementById('hit-template').innerHTML,
            empty: `We didn't find any results for the search <em>"{{query}}"</em>`,
            },
        })
    ]);

// search.addWidgets([
//     instantsearch.widgets.searchBox({
//       container: "#searchbox"
//     })
// ]);

// search.addWidgets([
//     instantsearch.widgets.hits({
//       container: "#hits"
//     })
// ]);
  
search.start();
