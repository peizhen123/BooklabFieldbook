<p>This week&rsquo;s lab seems fairly familiar to me. One reason is that genre classification has been one of the most important aspect both in Digital Humanities and in Humanities in general. The other is that I was actually involved in several projects that are related to genre classification, and in fact met quite a few difficulties.</p>
<p>Several years ago, I was involved in a Chinese Periodicals corpus collected in Chicago Textual Lab under the supervision of Professor Hoyt Long. We were looking at a Republican China Periodicals Corpus (1918-1949), and one task is to differentiate between fiction and nonfiction. Seemingly easy because of the binary sense, it turns out very difficult to reach an accuracy of over 90%. This is quite surprising, as there is a 50% accuracy even by random chance since we had only two categories. We ended up using three methods in combination: human defined features related to fiction such as quotation marks; LDA topic modeling; and a ML model inspired by Keras and TensorFlow. There are many reasons that account for the difficulty, one is the difficulty of word segmentation for Chinese characters. We are targeting Republican China, a time when Ancient Chinese was slowly transitioning towards modern Chinese. This makes dominant segmentation tools like Jieba not that accurate since they are more based on internet-collected modern Chinese.</p>
<p>I am not going to dwell on the unrelated difficulties, but our lab today, especially the second one, gives me a chance to ponder on if the category of fiction-nonfiction itself adds more difficulties. Our second topic modeling classification has six categories, literary, info, news, ads, poetry and recipe, and all six categories seem to be quite clearly defined through particular themes. This is especially true for recipe, ads and news. In the notes it is also noted, &ldquo;Looking at the actual numbers, we see that the most accurate classifiers seem to be for ads and news, while the least accurate is for "information"&mdash;which is, to be fair, a pretty baggy genre, as we can discuss together.&rdquo; Then is fiction and nonfiction also baggy genre classifications? At first glance, the boundaries between fiction and nonfiction is very clear &ndash; &ldquo;the imagination, not presented as fact&rdquo; is fiction, and the rest is nonfiction. While the boundary is clear, it does not mean it is easy to apply to computational models, since in real cases, we, as humans, don&rsquo;t seem to have a clear rule to decide when we see a fiction or a nonfiction. If the roles are not clear for humans, it will be more difficult for computers to differentiate.</p>
<p>Of course, now we have deep learning models to train data and classify different categories through the black box, that is, humans will not know what rules computers are using (just like the third method we used for the Republican Chinese Periodicals). I understand the purpose of genre classification is usually for extracting a particular dataset (as said in the Colab note, &ldquo;What this post helps us understand is one use case for genre classification, which is when you have an enormous dataset and want to find one specific kind of material within it&rdquo;), and <em>how</em> computers classify texts don&rsquo;t always matter as long as accuracy is reached. But as a literary scholar, the accuracy is uninterpretable, rendering the results insignificant if we are viewing from a literary perspective.</p>
<p>With this concern, I found the &ldquo;Visualizing Multiple Possibilities&rdquo; section of the second lab especially helpful. Instead of treating genre as a definite term, the visualization helps us understand genre as nonbinary, indefinite, and even fluid. Out of curiosity, I looked at the original text of id 8592469340 (lower left of the image), because it has several elements (recipe, literary, poetry, and info) at the same time. it turns out that it is a poetic work with quite a few mentioning of ingredients and taste (like sugar), while also indicating some people and places&rsquo; names. This suggests that a combination of different classification methods (besides topic modeling) might help increase the accuracy of classification. It also indicates the rather inclusive genre feature of poetry and literary (which usually are defined by forms rather by theme), and the possibility of genre crossover (poetry and literary can happen at the same time).</p>