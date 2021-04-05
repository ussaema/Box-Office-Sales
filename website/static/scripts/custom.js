$(window).on('load', function() {
    var status = $('#status');
    var preloader = $('#preloader');
    var body = $('body');
    status.fadeOut();
    preloader.delay(0).fadeOut('fast');
    body.delay(0).css({
        'overflow': 'visible'
    });
    var vidDefer = document.getElementsByTagName('iframe');
    for (var i = 0; i < vidDefer.length; i++) {
        if (vidDefer[i].getAttribute('data-src')) {
            vidDefer[i].setAttribute('src', vidDefer[i].getAttribute('data-src'));
        }
    }
})
$(function() {
    'use strict';
    var windowWidth = $(window).width();
    if (windowWidth > 1024) {
        var dropdown = $('.dropdown');
        dropdown.hover(function() {
            $(this).children('.dropdown-menu').fadeIn(300);
        }, function() {
            $(this).children('.dropdown-menu').fadeOut(300);
        });
    } else {
        var dropdownClick = $('.navbar a.dropdown-toggle');
        dropdownClick.on('click', function(e) {
            var $el = $(this);
            var $parent = $(this).offsetParent(".dropdown-menu");
            var $open = $('.nav li.open');
            $(this).parent("li").toggleClass('open');
            if (!$parent.parent().hasClass('nav')) {
                $el.next().css({
                    "top": $el[0].offsetTop,
                    "left": $parent.outerWidth() - 4
                });
            }
            $open.not($(this).parents("li")).removeClass("open");
            return false;
        });
    }
    var clickMenubtn = $('#nav-icon1');
    clickMenubtn.on('click', function() {
        $(this).toggleClass('open');
    });
    var tabsClick = $('.tabs .tab-links a, .tab-links-2 a, .tab-links-3 a');
    var multiItem = $('.slick-multiItem');
    var multiItem2 = $('.slick-multiItem2');
    tabsClick.on('click', function(e) {
        var currentAttrValue = $(this).attr('href');
        var tabsCurrent = $('.tabs ' + currentAttrValue);
        tabsCurrent.show().siblings().hide();
        $(this).parent('li').addClass('active').siblings().removeClass('active');
        e.preventDefault();
        multiItem.slick('setPosition');
        multiItem2.slick('setPosition');
    });

    function getTimeRemaining(endtime) {
        var t = Date.parse(endtime) - Date.parse(new Date());
        var seconds = Math.floor((t / 1000) % 60);
        var minutes = Math.floor((t / 1000 / 60) % 60);
        var hours = Math.floor((t / (1000 * 60 * 60)) % 24);
        var days = Math.floor(t / (1000 * 60 * 60 * 24));
        return {
            'total': t,
            'days': days,
            'hours': hours,
            'minutes': minutes,
            'seconds': seconds
        };
    }

    function initializeClock(id, endtime) {
        var clock = document.getElementById(id);
        if (clock != null) {
            var daysSpan = clock.querySelector('.days');
            var hoursSpan = clock.querySelector('.hours');
            var minutesSpan = clock.querySelector('.minutes');
            var secondsSpan = clock.querySelector('.seconds');
            var updateClock = function() {
                var t = getTimeRemaining(endtime);
                daysSpan.innerHTML = t.days;
                hoursSpan.innerHTML = ('0' + t.hours).slice(-2);
                minutesSpan.innerHTML = ('0' + t.minutes).slice(-2);
                secondsSpan.innerHTML = ('0' + t.seconds).slice(-2);
                if (t.total <= 0) {
                    clearInterval(timeinterval);
                }
            }
            updateClock();
            var timeinterval = setInterval(updateClock, 1000);
        }
    }
    var deadline = new Date(Date.parse(new Date()) + 25 * 24 * 60 * 60 * 1000);
    initializeClock('clockdiv', deadline);
    var tweets = jQuery(".tweet");
    jQuery(tweets).each(function(t, tweet) {
        var id = jQuery(this).attr('id');
        twttr.widgets.createTweet(id, tweet, {
            conversation: 'none',
            cards: 'hidden',
            linkColor: 'default',
            theme: 'light'
        });
    });
    multiItem2.slick({
        infinite: true,
        slidesToShow: 4,
        slidesToScroll: 1,
        rows: 1,
        arrows: false,
        dots: true,
        responsive: [{
            breakpoint: 1200,
            settings: {
                rows: 1,
                slidesToShow: 3,
                slidesToScroll: 1,
                infinite: true,
                dots: true
            }
        }, {
            breakpoint: 768,
            settings: {
                rows: 1,
                slidesToShow: 3,
                slidesToScroll: 1
            }
        }, {
            breakpoint: 480,
            settings: {
                rows: 1,
                slidesToShow: 1,
                slidesToScroll: 1
            }
        }]
    });
    multiItem.slick({
        infinite: true,
        slidesToShow: 4,
        slidesToScroll: 4,
        arrows: false,
        draggable: true,
        dots: true,
        responsive: [{
            breakpoint: 1024,
            settings: {
                slidesToShow: 3,
                slidesToScroll: 3,
                infinite: true,
                dots: true
            }
        }, {
            breakpoint: 768,
            settings: {
                slidesToShow: 2,
                slidesToScroll: 2
            }
        }, {
            breakpoint: 480,
            settings: {
                slidesToShow: 1,
                slidesToScroll: 1
            }
        }]
    });
    var multiItemSlider = $('.slick-multiItemSlider');
    multiItemSlider.slick({
        infinite: true,
        slidesToShow: 4,
        slidesToScroll: 4,
        arrows: false,
        draggable: true,
        autoplay: true,
        autoplaySpeed: 4200,
        dots: true,
        responsive: [{
            breakpoint: 1024,
            settings: {
                slidesToShow: 3,
                slidesToScroll: 3,
                infinite: true,
                dots: true
            }
        }, {
            breakpoint: 768,
            settings: {
                slidesToShow: 2,
                slidesToScroll: 2
            }
        }, {
            breakpoint: 480,
            settings: {
                slidesToShow: 1,
                slidesToScroll: 1
            }
        }]
    });
    var singleItem = $('.slider-single-item');
    singleItem.slick({
        infinite: true,
        slidesToShow: 1,
        slidesToScroll: 1,
        arrows: true,
        dots: true,
        draggable: true,
        responsive: [{
            breakpoint: 1024,
            settings: {
                slidesToShow: 1,
                slidesToScroll: 1,
                infinite: true,
                dots: true,
                arrows: true
            }
        }, {
            breakpoint: 768,
            settings: {
                slidesToShow: 1,
                slidesToScroll: 1
            }
        }, {
            breakpoint: 480,
            settings: {
                slidesToShow: 1,
                slidesToScroll: 1,
                arrows: false,
            }
        }]
    });
    var slickTw = $('.slick-tw');
    slickTw.slick({
        infinite: true,
        slidesToShow: 1,
        slidesToScroll: 1,
        dots: true,
        draggable: true,
        arrows: false,
        responsive: [{
            breakpoint: 1024,
            settings: {
                slidesToShow: 1,
                slidesToScroll: 1,
                infinite: true,
                dots: true,
                arrows: false
            }
        }, {
            breakpoint: 768,
            settings: {
                slidesToShow: 1,
                slidesToScroll: 1,
            }
        }, {
            breakpoint: 480,
            settings: {
                slidesToShow: 1,
                slidesToScroll: 1,
                arrows: false,
            }
        }]
    });
    var slidefor = $('.slider-for');
    var slidenav = $('.slider-nav');
    slidefor.slick({
        slidesToShow: 1,
        slidesToScroll: 1,
        arrows: false,
        fade: true,
        asNavFor: '.slider-nav',
    });
    slidenav.slick({
        slidesToShow: 5,
        slidesToScroll: 1,
        asNavFor: '.slider-for',
        dots: true,
        focusOnSelect: true,
        responsive: [{
            breakpoint: 1024,
            settings: {
                slidesToShow: 3,
                slidesToScroll: 3,
                infinite: true,
                arrows: true
            }
        }, {
            breakpoint: 768,
            settings: {
                slidesToShow: 3,
                slidesToScroll: 3
            }
        }, {
            breakpoint: 480,
            settings: {
                slidesToShow: 1,
                slidesToScroll: 1,
                arrows: true
            }
        }]
    });
    var slidefor2 = $('.slider-for-2');
    var slidenav2 = $('.slider-nav-2');
    slidefor2.slick({
        slidesToShow: 1,
        slidesToScroll: 1,
        arrows: false,
        fade: true,
        asNavFor: '.slider-nav-2',
    });
    slidenav2.slick({
        slidesToShow: 3,
        slidesToScroll: 1,
        asNavFor: '.slider-for-2',
        dots: false,
        arrows: true,
        focusOnSelect: true,
        vertical: true,
    });
    var fancyboxmedia = $('.fancybox-media');
    fancyboxmedia.fancybox({
        openEffect: 'float',
        closeEffect: 'none',
        helpers: {
            media: {},
            overlay: {
                locked: false
            }
        }
    });
    fancyboxmedia.attr('rel', 'playlist').fancybox({
        openEffect: 'none',
        closeEffect: 'none',
        prevEffect: 'none',
        nextEffect: 'none',
        helpers: {
            media: {}
        },
        youtube: {
            autoplay: 1,
            hd: 1,
            wmode: 'opaque',
            vq: 'hd720'
        }
    });
    var imglightbox = $(".img-lightbox");
    imglightbox.fancybox({
        helpers: {
            title: {
                type: 'float'
            },
            overlay: {
                locked: false
            }
        }
    });
    imglightbox.fancybox({
        afterShow: function() {
            var gallerySize = this.group.length,
                next, prev;
            if (this.index == gallerySize - 1) {
                next = imglightbox.eq(0).attr("title"), prev = imglightbox.eq(this.index - 1).attr("title");
            } else if (this.index == 0) {
                next = imglightbox.eq(this.index + 1).attr("title"), prev = imglightbox.eq(gallerySize - 1).attr("title");
            } else {
                next = imglightbox.eq(this.index + 1).attr("title"), prev = imglightbox.eq(this.index - 1).attr("title");
            }
            var lightboxnext = $(".img-lightbox-next");
            var lightboxprev = $(".img-lightbox-prev");
            lightboxnext.attr("title", next);
            lightboxprev.attr("title", prev);
        }
    });
    var loginLink = $(".loginLink");
    var signupLink = $(".signupLink");
    var loginct = $("#login-content");
    var signupct = $("#signup-content");
    var loginWrap = $(".login-wrapper");
    var overlay = $(".overlay");
    loginWrap.each(function() {
        $(this).wrap('<div class="overlay"></div>')
    });
    loginLink.on('click', function(event) {
        event.preventDefault();
        loginct.parents(overlay).addClass("openform");
        $(document).on('click', function(e) {
            var target = $(e.target);
            if ($(target).hasClass("overlay")) {
                $(target).find(loginct).each(function() {
                    $(this).removeClass("openform");
                });
                setTimeout(function() {
                    $(target).removeClass("openform");
                }, 350);
            }
        });
    });
    signupLink.on('click', function(event) {
        event.preventDefault();
        signupct.parents(overlay).addClass("openform");
        $(document).on('click', function(e) {
            var target = $(e.target);
            if ($(target).hasClass("overlay")) {
                $(target).find(signupct).each(function() {
                    $(this).removeClass("openform");
                });
                setTimeout(function() {
                    $(target).removeClass("openform");
                }, 350);
            }
        });
    });
    var closebt = $(".close");
    closebt.on('click', function(e) {
        e.preventDefault();
        var overlay = $(".overlay");
        overlay.removeClass("openform");
    });
    var multiselect = $(".ui.fluid.dropdown");
    multiselect.dropdown({
        allowLabels: true
    })
    multiselect.dropdown({
        'set selected': 'Role1,Role2'
    });
    $(window).scroll(function(event) {
        var scrollPos = $(window).scrollTop(),
            header = $('header');
        if (scrollPos > 300) {
            header.addClass('sticky');
        } else {
            header.removeClass('sticky');
        }
    });
    var backtotop = $('#back-to-top');
    backtotop.on('click', function(e) {
        e.preventDefault();
        $('html,body').animate({
            scrollTop: 0
        }, 700);
    });
    var scrolldownlanding = $('#discover-now');
    scrolldownlanding.on('click', function(e) {
        e.preventDefault();
        $('html,body').animate({
            scrollTop: $(document).height() - $(window).height()
        }, 700);
    });
    if (windowWidth > 1200) {
        var stickySidebar = $('.sticky-sb');
        var mainCt = $('.main-content');
        if (stickySidebar.length > 0) {
            var stickyHeight = stickySidebar.height(),
                sidebarTop = stickySidebar.offset().top;
        }
        $(window).scroll(function() {
            if (stickySidebar.length > 0) {
				
                var scrollTop = $(window).scrollTop();
                if (sidebarTop < scrollTop) {
                    stickySidebar.css('top', scrollTop - sidebarTop -80);
                    var sidebarBottom = stickySidebar.offset().top + stickyHeight,
                        stickyStop = mainCt.offset().top + mainCt.height() -130;
                    if (stickyStop < sidebarBottom) {
                        var stopPosition = mainCt.height() - stickyHeight - 130;
                        stickySidebar.css('top', stopPosition);
                    }
                } else {
                    stickySidebar.css('top', 0);
                }
            }
        });
        $(window).resize(function() {
            if (stickySidebar.length > 0) {
                stickyHeight = stickySidebar.height();
            }
        });
    }
});