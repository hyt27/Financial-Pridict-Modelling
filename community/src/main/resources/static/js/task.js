
$(function(){
    $("#submitBtn").click(submit_task);
    $(".close").click(delete_msg);

    $("#viewResultLink").click(view_result)
});


function submit_task() {
    $("#submitmodel").modal("hide");

    // var toName = $("#recipient-name").val();
    // var content = $("#message-text").val();
    var category = $("#category").val();
    var stock = $("#stock").val();
    var startDate = $("#start-date").val();
    var endDate = $("#end-date").val();
    var model = $("#model").val();
    var strategy = $("#strategy").val();
    var initial = $("#initial").val();

    if (!category || !stock || !startDate || !endDate || !model || !strategy || !initial) {
        alert("Please fill out all fields before submitting.");
        return;  // 中止函数执行
    }

    var formData = {
        category: category,
        stock: stock,
        startDate: startDate,
        endDate: endDate,
        model: model,
        strategy: strategy,
        initial: initial
    };
    console.log("Form Data Submitted: ", formData);

    $.post(
        CONTEXT_PATH + "/addTask",
        {
            "category": category, "model": model, "strategy": strategy, "initial": initial,
            "stock": stock, "startDate": startDate, "endDate": endDate
        },
        function (data) {
            data = $.parseJSON(data);
            if (data.code == 0) {
                $("#hintBody").text("Submitted!");
            } else {
                $("#hintBody").text(data.msg);
            }

            $("#hintModal").modal("show");
            setTimeout(function () {
                $("#hintModal").modal("hide");
                location.reload();
            }, 2000);
        }

    );
}

// task.js
function view_result() {

}






