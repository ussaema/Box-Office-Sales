$(document).ready(function () {
    console.log("loaded");
    /**
     * Create Book
     */
    // process the form
    $('#new_user_form').submit(function (e) {
        // console.log("Creating the book");
        e.preventDefault();
        // get the form data
       
        $.ajax({
            type: 'POST',
            url: 'create/',
            data: {
				name:$('#name').val(),
				csrfmiddlewaretoken:$('input[name=csrfmiddlewaretoken]').val()
			},
			sucess:function(){
				alert('sdsd');
			}
        });
    });
});

