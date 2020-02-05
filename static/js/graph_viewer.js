
var graph_space_pos_args = []
var graph_space_neg_args = []
var graph_space_rating = []
var currentPrior = [];
var currentLiklihood = []
var observations = [];

function generateUniformPrior(){

    $("div.prior").remove();
    $("div.liklihood").remove();
    $("div.posterior").remove();
    $("graph-image").remove();

    pos_args_string = document.getElementById('graph_space_pos_args').value;
    pos_args = pos_args_string.split(',');

    neg_args_string = document.getElementById('graph_space_neg_args').value;
    neg_args = neg_args_string.split(',');

    rating_string = document.getElementById('graph_space_rating').value;
    rating = parseInt(rating_string);

    graph_space_pos_args = pos_args;
    graph_space_neg_args = neg_args;
    graph_space_rating = rating;

    all_args = pos_args.concat(neg_args);
    graph_label = all_args.join('_');
    var location = $SCRIPT_ROOT + 'static/graphs/';

    var graph_location = location + graph_label;

    $.get($SCRIPT_ROOT + '/generateGraphSpacewithUniformPrior', {pos_args: JSON.stringify(pos_args), 
                                                            neg_args: JSON.stringify(neg_args),
                                                            rating: rating}, 
                                                            success=function(prior_distribution) {

                    console.log("passing the values");
                    currentPrior = prior_distribution;

                    $('#distribution').append('<div class="row" id="update-table-headers"></div>');                                                         
                    $('#update-table-headers').append('<div class="col-sm-2">No</div>'); 
                    $('#update-table-headers').append('<div class="col-sm-4">G</div>'); 
                    $('#update-table-headers').append('<div class="col-sm-2">P(G)</div>'); 
                    $('#update-table-headers').append('<div class="col-sm-2">P(T|G)</div>'); 
                    $('#update-table-headers').append('<div class="col-sm-2">P(G|T)</div>'); 

                    for (i = 0; i < prior_distribution.length; i++) {
                    var image_url = graph_location + '_' + i + '.jpeg';

                    $('#distribution').append('<div class="row" style="margin: 1px 1px 1px 1px; border-style: solid; border-width: 1px;" id="' + 'all_data_' + i + '"></div>');
                    $('#all_data_' + i).append('<div class="col-sm-2">'+ i +'</div>');
                    $('#all_data_' + i).append('<div class="col-sm-4">'+ '<img src="' + image_url + '" style="    max-width:50%; height:auto;" />' +'</div>');
                    $('#all_data_' + i).append('<div class="col-sm-2 prior">'+ currentPrior[i].toFixed(3) +'</div>');

                }
            }
        );
}


function generateGraphSpacewithPrior() {

    pos_args_string = document.getElementById('graph_space_pos_args').value;
    pos_args = pos_args_string.split(',');

    neg_args_string = document.getElementById('graph_space_neg_args').value;
    neg_args = neg_args_string.split(',');

    rating_string = document.getElementById('graph_space_rating').value;
    rating = parseInt(rating_string);

    graph_space_pos_args = pos_args;
    graph_space_neg_args = neg_args;
    graph_space_rating = rating;

    // pos_args = ['a', 'b'];
    // neg_args = ['c'];
    // rating = 10;

    all_args = pos_args.concat(neg_args);
    graph_label = all_args.join('_');

    var location = $SCRIPT_ROOT + 'static/graphs/';

    var graph_location = location + graph_label;

    $.get($SCRIPT_ROOT + '/generateGraphSpacewithPrior', {pos_args: JSON.stringify(pos_args), 
                                                                    neg_args: JSON.stringify(neg_args),
                                                                    rating: rating}, 
                                                                    success=function(prior_distribution) {
            console.log("passing the values");
            currentPrior = prior_distribution;

            $('#distribution').append('<div class="row" id="update-table-headers"></div>');                                                         
            $('#update-table-headers').append('<div class="col-sm-2">No</div>'); 
            $('#update-table-headers').append('<div class="col-sm-4">G</div>'); 
            $('#update-table-headers').append('<div class="col-sm-2">P(G)</div>'); 
            $('#update-table-headers').append('<div class="col-sm-2">P(T|G)</div>'); 
            $('#update-table-headers').append('<div class="col-sm-2">P(G|T)</div>'); 

            for (i = 0; i < prior_distribution.length; i++) {
                var image_url = graph_location + '_' + i + '.jpeg';

                $('#distribution').append('<div class="row" style="margin: 1px 1px 1px 1px; border-style: solid; border-width: 1px;" id="' + 'all_data_' + i + '"></div>');
                $('#all_data_' + i).append('<div class="col-sm-2">'+ i +'</div>');
                $('#all_data_' + i).append('<div class="col-sm-4 graph-image">'+ '<img src="' + image_url + '" style="    max-width:50%; height:auto;" />' +'</div>');
                $('#all_data_' + i).append('<div class="col-sm-2 prior">'+ currentPrior[i].toFixed(3) +'</div>');

            }
        }
    );

}

function addObservation() {

    pos_args_string = document.getElementById('observation_pos_args').value;
    pos_args = pos_args_string.split(',');

    neg_args_string = document.getElementById('observation_neg_args').value;
    neg_args = neg_args_string.split(',');

    rating_string = document.getElementById('observation_rating').value;
    rating = parseInt(rating_string);

    var attacks = []
    attacks_string = document.getElementById('observation_attacks').value;

    if (attacks_string != ''){
        attacks_string = attacks_string.split(",");
        for (i = 0; i < attacks_string.length; i++){
            var arguments = attacks_string[i].split("-");
            attacks.push([arguments[0],arguments[1]]);
        }
    }

    // pos_args = ['a'];
    // neg_args = ['c'];
    // rating = 10;
    // attacks = [['a','c']];

    var observation = {};
    observation['pos_args'] = pos_args;
    observation['neg_args'] = neg_args;
    observation['rating'] = rating;
    observation['attacks'] = attacks;
    observations.push(observation);

    var observationNo = observations.length - 1;
    var location = $SCRIPT_ROOT + 'static/graphs/';
    var graph_label = location + 'observation_' + observationNo;

    $.get($SCRIPT_ROOT + '/generateObservationGraph', {pos_args: JSON.stringify(pos_args), 
                                                            neg_args: JSON.stringify(neg_args),
                                                            rating: rating,
                                                            observationNo: observationNo,
                                                            attacks: JSON.stringify(attacks)}, 
                                                            success=function(graph_location) {
            
            var graph_label = location + 'observation_' + observationNo + '.jpeg';                                             

            $('#observations').append('<div class="row" style="margin: 1px 1px 1px 1px; border-style: solid; border-width: 1px;" id="' + 'observation_' + observationNo + '"></div>');
            $('#observation_' + observationNo).append('<div class="col-sm-2" style="text-align:center;">'+ '<p>#' + observationNo + '</p>');
            $('#observation_' + observationNo).append('<div class="col-sm-4">'+ '<img src="' + graph_label + '" style="    max-width:80%; height:auto;" />' +'</div>');
            $('#observation_' + observationNo).append('<div class="col-sm-2">'+ '<p>Rating: '+ rating + '</p>' +'</div>');  
            $('#observation_' + observationNo).append('<div class="col-sm-4">'+ '<button class="btn btn-primary" onclick="updatePrior(' + observationNo + ')">Update Distribution</button>' +'</div>');                                                
        }
    );
}

function updatePrior(observationNo) {

    $("div.prior").remove();
    $("div.liklihood").remove();
    $("div.posterior").remove();

    console.log(observationNo);

    var graphSpaceSummary = {};
    graphSpaceSummary['pos_args'] = graph_space_pos_args;
    graphSpaceSummary['neg_args'] = graph_space_neg_args;
    graphSpaceSummary['rating'] = graph_space_rating;

    $.get($SCRIPT_ROOT + '/generatePosteriorDistributionWithObsevation', {currentPrior: JSON.stringify(currentPrior),
                                                            pos_args: JSON.stringify(observations[observationNo]['pos_args']), 
                                                            neg_args: JSON.stringify(observations[observationNo]['neg_args']),
                                                            rating: observations[observationNo]['rating'],
                                                            attacks: JSON.stringify(observations[observationNo]['attacks']),
                                                            observationNo: observationNo,
                                                            graphSpaceSummary: JSON.stringify(graphSpaceSummary)},
                                                            success=function(distributions) {

            var liklihood_distribution = distributions['liklihood_distribution'];
            var posterior_distribution = distributions['posterior_distribution'];

            for (i = 0; i < currentPrior.length; i++) {
                $('#all_data_' + i).append('<div class="col-sm-2 prior">'+ currentPrior[i].toFixed(3) +'</div>');
                $('#all_data_' + i).append('<div class="col-sm-2 liklihood">'+ liklihood_distribution[i].toFixed(3) +'</div>');
                $('#all_data_' + i).append('<div class="col-sm-2 posterior">'+ posterior_distribution[i].toFixed(3) +'</div>');
            } 
            
            console.log(liklihood_distribution);

            // Change the posterior so we are ready for another update
            currentPrior = posterior_distribution;
        }
    );
}

