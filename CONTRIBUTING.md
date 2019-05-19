# Contributing to gpirt

To contribute code to this repository, please use the following process:
  - Submit an issue describing the problem to fix or enhancement to add
  - Fork the repository
  - Make your changes in a new branch:
  
      ```
      git checkout -b my-new-feature master
      ```
      
  - Add your code, committing often and **following the commit message and
    coding style rules below**
  - Push your new branch to GitHub:
  
      ```
      git push origin my-new-feature
      ```
      
  - Submit a pull request

## Commit message style

Every commit should have a commit message that follows these rules:

  - The commit message is both sufficiently descriptive and concise
  - The summary (first line) is about 50 characters or less
  - The summary's first word is capitalized (the summary is not in title case)
  - The summary is in the imperative; e.g., "Add new function"
  - Where helpful, additional explanation is added on additional lines
    with one blank line in between the summary and additional description
  - Additional description lines **must** be no longer than 72 characters
    
This commit message style was popularized by Tim Pope; you can read more
about it here:
https://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html

## Coding style

To keep the coding style uniform, please follow the following style rules:

  - Indents are four spaces, not two as is popular in some R circles
  - Use braces for control flow blocks, even if one line
  - Use snake_case, not CamelCase
  - Typically place spaces around operators

Here is an example demonstrating the above rules:

```r
example_function <- function(x = 2) {
    if ( x < 1 ) {
        return(x)
    }
    return(log(x) * 2 + 1)
}
```

-----
<sub>These contributing guidelines are also used by [`gpmlr`](https://github.com/duckmayr/gpmlr)
